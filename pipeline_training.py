import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from prefect import flow, task
from typing import Tuple, Dict, List
from models import logger, UNetWithAttention, FusionModel
from data_util import GridFileProcessor, GridDataset, collate_variable_size

# All Training Tasks

@task(name="Fetch Dataset")
def fetch_training_dataset(random_data=False):
    grids, targets = [], []
    if random_data:
        pass
        # Generate data
        # grids, targets = generate_data(n_samples, min_size, max_size)
    else:
        all_data = (GridFileProcessor("dataset/"+fname).create_input_output() for fname in os.listdir("dataset/")  if fname.startswith("training_"))
        for _grids, _targets in all_data:
            grids.extend(_grids)
            targets.extend(_targets)
    return grids, targets

@task(name="Prepare DataLoaders")
def prepare_dataloaders(grids: List[np.ndarray], targets: List[np.ndarray], 
                       batch_size=8) -> Dict[str, DataLoader]:
    """Split data and create dataloaders"""
    logger.info("Preparing train/val dataloaders")
    
    n = len(grids)
    split = int(0.8 * n)
    
    train_dataset = GridDataset(grids[:split], targets[:split], augment=True)
    val_dataset = GridDataset(grids[split:], targets[split:], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2, collate_fn=collate_variable_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2, collate_fn=collate_variable_size)
    
    return {"train": train_loader, "val": val_loader}


@task(name="Train Route A: Pretrain CNN")
def train_route_a_pretrain(dataloaders: Dict[str, DataLoader], 
                          epochs=20, lr=1e-3, device='cuda') -> nn.Module:
    """Route A: Pretrain CNN encoder-decoder without attention"""
    logger.info("Route A: Pretraining CNN without attention")
    
    model = UNetWithAttention().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in dataloaders['train']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x, use_attention=False)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_loss = train_loss / n_batches
        logger.info(f"Route A Pretrain - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


@task(name="Train Route A: Attention Only")
def train_route_a_attention(model: nn.Module, dataloaders: Dict[str, DataLoader],
                           epochs=15, lr=1e-4, device='cuda') -> nn.Module:
    """Route A: Fix CNN and train attention only"""
    logger.info("Route A: Training attention with fixed CNN")
    
    # Freeze encoder and decoder conv blocks
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    for name, param in model.decoder.named_parameters():
        if 'attn' not in name:
            param.requires_grad = False
    
    # Only train attention modules
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in dataloaders['train']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x, use_attention=True)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_loss = train_loss / n_batches
        logger.info(f"Route A Attention - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Unfreeze for inference
    for param in model.parameters():
        param.requires_grad = True
    
    torch.save(model.state_dict(), 'saved_models/model_a.pth')
    return model


@task(name="Train Route B: End-to-End")
def train_route_b_e2e(dataloaders: Dict[str, DataLoader],
                     epochs=30, lr=1e-3, device='cuda') -> nn.Module:
    """Route B: End-to-end training with attention"""
    logger.info("Route B: End-to-end training with attention")
    
    model = UNetWithAttention().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in dataloaders['train']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x, use_attention=True)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_loss = train_loss / n_batches
        scheduler.step(avg_loss)
        logger.info(f"Route B E2E - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), 'saved_models/model_b.pth')
    return model


@task(name="Generate Predictions")
def generate_predictions(model: nn.Module, dataloaders: Dict[str, DataLoader],
                        device='cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate predictions for fusion training"""
    logger.info("Generating predictions for fusion")
    
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloaders['train']:
            batch_x = batch_x.to(device)
            pred = model(batch_x, use_attention=True)
            preds.append(pred.cpu())
            targets.append(batch_y)
    
    return torch.cat(preds), torch.cat(targets)


@task(name="Train Fusion Model")
def train_fusion_model(preds_a: torch.Tensor, preds_b: torch.Tensor,
                      targets: torch.Tensor, epochs=10, lr=1e-3,
                      device='cuda') -> nn.Module:
    """Train fusion model to combine both routes"""
    logger.info("Training fusion model")
    
    fusion_model = FusionModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr)
    
    # Create simple dataloader
    dataset = torch.utils.data.TensorDataset(preds_a, preds_b, targets)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for epoch in range(epochs):
        fusion_model.train()
        train_loss = 0
        n_batches = 0
        
        for pred_a, pred_b, target in loader:
            pred_a = pred_a.to(device)
            pred_b = pred_b.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            fused = fusion_model(pred_a, pred_b)
            loss = criterion(fused, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_loss = train_loss / n_batches
        logger.info(f"Fusion - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    torch.save(fusion_model.state_dict(), 'saved_models/fusion_model.pth')
    return fusion_model

@task(name="Evaluate Models")
def evaluate_models(model_a: nn.Module, model_b: nn.Module,
                   fusion_model: nn.Module, dataloaders: Dict[str, DataLoader],
                   device='cuda'):
    """Evaluate all models on validation set"""
    logger.info("Evaluating models")
    
    model_a.eval()
    model_b.eval()
    fusion_model.eval()
    
    total_loss_a = 0
    total_loss_b = 0
    total_loss_fused = 0
    criterion = nn.BCEWithLogitsLoss()
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloaders['val']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred_a = model_a(batch_x, use_attention=True)
            pred_b = model_b(batch_x, use_attention=True)
            pred_fused = fusion_model(pred_a, pred_b)
            
            total_loss_a += criterion(pred_a, batch_y).item()
            total_loss_b += criterion(pred_b, batch_y).item()
            total_loss_fused += criterion(pred_fused, batch_y).item()
            n_batches += 1
    
    logger.info(f"Validation Results:")
    logger.info(f"  Route A Loss: {total_loss_a/n_batches:.4f}")
    logger.info(f"  Route B Loss: {total_loss_b/n_batches:.4f}")
    logger.info(f"  Fused Loss: {total_loss_fused/n_batches:.4f}")


# MAIN PREFECT TRAINING FLOW 

@flow(name="2D Grid Segmentation Pipeline")
def grid_segmentation_pipeline(device='cuda', random_data=False):
    """
    Main Prefect flow orchestrating the entire ML pipeline
    """
    logger.info("Starting 2D Grid Segmentation Pipeline")
    
    grids, targets = fetch_training_dataset.submit(random_data)
    dataloaders = prepare_dataloaders.submit(grids, targets, wait_for=[grids, targets])
    
    # Route A: Pretrain then fine-tune attention
    logger.info("Starting Route A (Pretrain + Attention)")
    model_a_pretrain = train_route_a_pretrain.submit(dataloaders, epochs=20, device=device, wait_for=[dataloaders])
    model_a = train_route_a_attention.submit(model_a, dataloaders, epochs=15, device=device, wait_for=[model_a_pretrain])
    
    # Route B: End-to-end training (runs in parallel conceptually)
    logger.info("Starting Route B (End-to-End)")
    model_b = train_route_b_e2e.submit(dataloaders, epochs=30, device=device, wait_for=[dataloaders])
    
    preds_a = generate_predictions.submit(model_a, dataloaders, device=device, wait_for=[model_a])
    preds_b = generate_predictions.submit(model_b, dataloaders, device=device, wait_for=[model_b])
    
    # Extract predictions and targets
    preds_a_tensor = preds_a.result()[0]
    targets_train = preds_a.result()[1]
    preds_b_tensor = preds_b.result()[0]
    
    # Train fusion model (waits for both prediction tasks)
    fusion_model = train_fusion_model.submit(
        preds_a_tensor, 
        preds_b_tensor, 
        targets_train, 
        epochs=10, 
        device=device,
        wait_for=[preds_a, preds_b]
    )
    
    # Evaluate all models (waits for all training to complete)
    evaluate_models.submit(
        model_a, 
        model_b, 
        fusion_model, 
        dataloaders, 
        device=device,
        wait_for=[model_a, model_b, fusion_model]
    )
    
    logger.info("Pipeline completed successfully")
    
    return {
        'model_a': model_a.result(),
        'model_b': model_b.result(),
        'fusion_model': fusion_model.result()
    }


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Run the pipeline
    models = grid_segmentation_pipeline(device=device,)
    