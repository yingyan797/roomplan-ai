import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from prefect import flow, task
from typing import Tuple, Dict, List
from models import logger

# ALL INFERENCE TASKS

@task(name="Prepare Batch Inference Input")
def prepare_batch_inference_input(grids_onehot: List[np.ndarray], device='cuda') -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Prepare multiple grids for batch inference
    
    Args:
        grids_onehot: List of 3D numpy arrays [H, W, 6] with one-hot encoded categories
        
    Returns:
        Tuple of (batched_tensor, list_of_original_shapes)
    """
    original_shapes = [(g.shape[0], g.shape[1]) for g in grids_onehot]
    
    # Find max dimensions needed for batch
    max_h = max((g.shape[0] + 7) // 8 * 8 for g in grids_onehot)
    max_w = max((g.shape[1] + 7) // 8 * 8 for g in grids_onehot)
    
    batch_tensors = []
    for grid in grids_onehot:
        # Pad to max batch size
        padded_grid = np.zeros((max_h, max_w, 6), dtype=grid.dtype)
        padded_grid[:grid.shape[0], :grid.shape[1], :] = grid
        
        # Convert to [C, H, W]
        grid_tensor = torch.from_numpy(padded_grid.transpose(2, 0, 1)).float()
        batch_tensors.append(grid_tensor)
    
    # Stack into batch [B, C, H, W]
    batched = torch.stack(batch_tensors)
    
    return batched, original_shapes


@task(name="Generate Batch Predictions")
def generate_batch_predictions(model: nn.Module, batch_tensor: torch.Tensor,
                               device='cuda') -> torch.Tensor:
    """Generate predictions for a batch of grids"""
    model.eval()
    
    with torch.no_grad():
        batch_tensor = batch_tensor.to(device)
        preds = model(batch_tensor, use_attention=True)
    
    return preds.cpu()

@task(name="Fuse Batch Predictions")
def fuse_batch_predictions(preds_a: torch.Tensor, preds_b: torch.Tensor,
                          fusion_model: nn.Module, device='cuda') -> torch.Tensor:
    """Fuse predictions from both routes for entire batch"""
    fusion_model.eval()
    
    with torch.no_grad():
        preds_a = preds_a.to(device)
        preds_b = preds_b.to(device)
        fused = fusion_model(preds_a, preds_b)
    
    return fused.cpu()

@task(name="Postprocess Batch Predictions")
def postprocess_batch_predictions(preds: torch.Tensor, original_shapes: List[Tuple[int, int]],
                                 threshold: float = 0.5) -> List[np.ndarray]:
    """
    Convert batch predictions to binary masks and crop to original sizes
    
    Args:
        preds: Prediction tensor [B, 1, H_padded, W_padded]
        original_shapes: List of (H_original, W_original) for each sample
        threshold: Threshold for binary mask
        
    Returns:
        List of binary masks, each [H_original, W_original]
    """
    masks = []
    
    # Convert to binary
    batch_masks = torch.sigmoid(preds) > threshold
    batch_masks = batch_masks.squeeze(1).numpy()  # [B, H, W]
    
    # Crop each mask to original size
    for i, (orig_h, orig_w) in enumerate(original_shapes):
        mask = batch_masks[i, :orig_h, :orig_w]
        masks.append(mask)
    
    return masks


# INFERENCE FLOW

def predict_single_grid(grid_onehot: np.ndarray, models: dict, device='cuda', threshold=0.5) -> np.ndarray:
    """
    Make predictions on a new grid of any size
    
    Args:
        grid_onehot: 3D numpy array [H, W, 6] with one-hot encoded categories
        models: Dictionary containing model_a, model_b, and fusion_model
        threshold: Threshold for binary mask
        
    Returns:
        Binary mask indicating positions for object 2
    """
    model_a = models['model_a'].eval()
    model_b = models['model_b'].eval()
    fusion_model = models['fusion_model'].eval()
    
    original_h, original_w = grid_onehot.shape[:2]
    
    # Pad to make dimensions divisible by 8
    h = ((original_h + 7) // 8) * 8
    w = ((original_w + 7) // 8) * 8
    
    padded_grid = np.zeros((h, w, 6), dtype=grid_onehot.dtype)
    padded_grid[:original_h, :original_w, :] = grid_onehot
    
    # Convert to [C, H, W] format and add batch dimension
    grid_tensor = torch.from_numpy(padded_grid.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        pred_a = model_a(grid_tensor, use_attention=True)
        pred_b = model_b(grid_tensor, use_attention=True)
        pred_fused = fusion_model(pred_a, pred_b)
        
        # Convert to binary mask
        mask = torch.sigmoid(pred_fused) > threshold
        mask = mask.squeeze().cpu().numpy()
    
    # Crop back to original size
    mask = mask[:original_h, :original_w]
    
    return mask

@flow(name="Batch Grid Inference")
def predict_batch_grids(grids_onehot: List[np.ndarray], models: dict,
                       device='cuda', threshold=0.5) -> List[np.ndarray]:
    """
    Prefect flow for making predictions on multiple grids with true batch processing
    
    Args:
        grids_onehot: List of 3D numpy arrays [H, W, 6] with one-hot encoded categories
        models: Dictionary containing model_a, model_b, and fusion_model
        threshold: Threshold for binary mask
        
    Returns:
        List of binary masks
    """
    logger.info(f"Starting batch inference flow for {len(grids_onehot)} grids")
    
    # Prepare all grids into a single batch tensor
    batch_future = prepare_batch_inference_input.submit(grids_onehot, device)
    batch_tensor, original_shapes = batch_future.result()
    
    # Generate predictions from Route A for entire batch
    pred_a_future = generate_batch_predictions.submit(
        models['model_a'],
        batch_tensor,
        device=device,
        wait_for=[batch_future]
    )
    
    # Generate predictions from Route B for entire batch (parallel with Route A)
    pred_b_future = generate_batch_predictions.submit(
        models['model_b'],
        batch_tensor,
        device=device,
        wait_for=[batch_future]
    )
    
    # Fuse predictions for entire batch (waits for both routes)
    fused_future = fuse_batch_predictions.submit(
        pred_a_future,
        pred_b_future,
        models['fusion_model'],
        device=device,
        wait_for=[pred_a_future, pred_b_future]
    )
    
    # Postprocess all predictions
    masks_future = postprocess_batch_predictions.submit(
        fused_future,
        original_shapes,
        threshold=threshold,
        wait_for=[fused_future]
    )
    
    masks = masks_future.result()
    
    logger.info(f"Batch inference completed for {len(masks)} grids")
    return masks
