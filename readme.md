** 2D Grid object placement machine learning pipeline

* Web Interface - 2D grid editor
Run app.py on localhost:5002, click on the canvas ro assign objects by layer
Data saved as JSON files under dataset/

* Main models used
UNet with Attention

* ML Training Pipeline using Prefect

Generate data/Fetch data from files
      ↓
Prepare dataloaders
      ↓
      ├──────(parallel)────┐
      ↓                    ↓
<Route A>               <Route B>
Pretrain CNN               ↓
      ↓              End-to-end Training
Train attention            ↓
      ↓                    ↓
Produce preds_a      Produce preds_b
      └─────────┬──────────┘
                ↓
        Train Fusion model
                ↓
            Evaluate


* ML Inference Pipeline

Prepare inference input
         ↓
    ├────────────┐
    ↓            ↓
pred_route_a  pred_route_b  (parallel)
    └────┬──────┘
         ↓
  Fuse predictions
         ↓
  Postprocess prediction