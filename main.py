"""
TOAM-YOLO Training and Evaluation Script

This script implements training and evaluation pipeline for TOAM-YOLO framework
based on the research paper: "TOAM-YOLO: A Lightweight Tiny Object-Aware 
Multi-Expert YOLO Framework for Diverse Domains"

Usage:
    python main.py --model_yaml <path_to_model_config> --data_yaml <path_to_dataset_config> 
                   --run_name <experiment_name> [additional_options]

Key Features:
- Automated training with TOAM-YOLO architecture
- Validation and test set evaluation
- Results logging to CSV for experiment tracking
- Configurable hyperparameters and training settings

Reference: TOAM-YOLO paper - Training methodology and evaluation protocol
"""

from ultralytics import YOLO
from IPython.display import display, Image, clear_output
import torch
import gc
import os
import pandas as pd 

# Set random seed for reproducibility
torch.manual_seed(42)

def train_model(model_path, yaml_path, run_name, project="runs/train", epochs=200, batch=16):
    """
    Train TOAM-YOLO model with specified configuration.
    
    Args:
        model_path (str): Path to model YAML configuration (e.g., toam-yolo.yaml)
        yaml_path (str): Path to dataset YAML configuration
        run_name (str): Unique identifier for this training run
        project (str): Base directory for saving training outputs
        epochs (int): Number of training epochs
        batch (int): Training batch size
        
    Returns:
        YOLO: Trained model instance
        
    Reference: TOAM-YOLO paper - Training configuration and hyperparameters
    """
    model = YOLO(model_path)
    model.train(
        data=yaml_path, 
        epochs=epochs, 
        imgsz=640,
        batch=batch, 
        name=run_name, 
        amp=False,  # Mixed precision training disabled for stability
        project=project,
        #save_period=50,  # Save checkpoint every 50 epochs
        # Data augmentation settings (commented - adjust based on dataset)
        # augment=True, hsv_h=0.015, hsv_s=0.2, hsv_v=0.6, 
        # mosaic=0.4, erasing=0.3, shear=0.0, scale=0.5, close_mosaic=15,
        patience=50,  # Early stopping patience
        optimizer="AdamW",  # Optimizer selection
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate
        cos_lr=False  # Cosine learning rate scheduler
    )
    return model

def val_results(model_path, yaml_path):
    """
    Evaluate model on validation set.
    
    Args:
        model_path (str): Path to trained model weights (.pt file)
        yaml_path (str): Path to dataset configuration
        
    Returns:
        Results object containing validation metrics
    """
    model = YOLO(model_path)
    results = model.val(data=yaml_path, imgsz=640, batch=8, device='cuda')
    print(f"Validation Results: {results}")
    return results

def test_results(model_path, yaml_path):
    """
    Evaluate model on test set.
    
    Args:
        model_path (str): Path to trained model weights (.pt file)
        yaml_path (str): Path to dataset configuration
        
    Returns:
        Results object containing test metrics
    """
    model = YOLO(model_path)
    results = model.val(data=yaml_path, imgsz=640, batch=8, device = 'cuda',split='test')
    print(f"Test Results: {results}")
    return results

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate TOAM-YOLO model")
    parser.add_argument("--model_yaml", type=str, required=True, 
                       help="Path to model YAML file (e.g., toam-yolo.yaml)")
    parser.add_argument("--data_yaml", type=str, required=True, 
                       help="Path to dataset YAML file (e.g., dataset/data.yaml)")
    parser.add_argument("--run_name", type=str, required=True, 
                       help="Unique name for this training run")
    parser.add_argument("--project", type=str, default="runs/train", 
                       help="Project directory for training outputs")
    parser.add_argument("--results_csv", type=str, default="results.csv", 
                       help="Path to save experiment results CSV")
    parser.add_argument("--epochs", type=int, default=200, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Training batch size")
    args = parser.parse_args()

    # Handle existing run name conflicts by appending suffix
    if not os.path.exists(os.path.join(args.project, args.run_name)):
        run_name = args.run_name
        model = train_model(args.model_yaml, args.data_yaml, args.run_name, 
                          args.project, args.epochs, args.batch_size)
    else:
        run_name = args.run_name + "_1"
        model = train_model(args.model_yaml, args.data_yaml, run_name, 
                          args.project, args.epochs, args.batch_size)

    # Evaluate trained model on validation and test sets
    torch.cuda.empty_cache()
    gc.collect()
    model_path = os.path.join(args.project, run_name, "weights", "best.pt")
    val = val_results(model_path, args.data_yaml).results_dict
    test = test_results(model_path, args.data_yaml).results_dict
    
    # Extract key metrics for logging
    # Standard YOLO metrics: precision, recall, mAP@0.5, mAP@0.5:0.95
    results = {
        "run_name": run_name,
        "precision": val['metrics/precision(B)'],
        "recall": val['metrics/recall(B)'],
        "mAP50": val['metrics/mAP50(B)'],
        "mAP50-95": val['metrics/mAP50-95(B)'],
        "test_precision": test['metrics/precision(B)'],
        "test_recall": test['metrics/recall(B)'],
        "test_mAP50": test['metrics/mAP50(B)'],
        "test_mAP50-95": test['metrics/mAP50-95(B)']
    }

    # Save results to CSV for experiment tracking
    df = pd.DataFrame([results])
    if os.path.exists(args.results_csv):
        df.to_csv(args.results_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(args.results_csv, index=False)
    
    print(f"Results saved to {args.results_csv}")
    print("Training and validation completed successfully.")