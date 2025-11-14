"""
Main Training Script for GNN Task-to-Resource Mapping
======================================================

This script orchestrates the complete training pipeline:
1. Load preprocessed graph data
2. Create train/validation/test splits
3. Initialize GNN model
4. Train with validation
5. Evaluate on test set
6. Save results and visualizations
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from torch_geometric.loader import DataLoader

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gnn_model import create_model, count_parameters
from src.train_gnn import GNNTrainer, evaluate_model


def load_dataset(data_dir='gnn_solution'):
    """
    Load preprocessed graph dataset
    
    Args:
        data_dir: Directory containing preprocessed data
        
    Returns:
        tuple: (train_data, val_data, test_data, dataset_info)
    """
    print(f"\n{'='*70}")
    print("Loading preprocessed graph dataset...")
    print(f"{'='*70}")
    
    # Get the script directory and construct absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Parent of Script/ is project root
    data_dir_abs = os.path.join(project_dir, data_dir)
    
    # Load train/test split (contains the graphs directly)
    split_path = os.path.join(data_dir_abs, 'train_test_split.pt')
    split_data = torch.load(split_path, weights_only=False)
    
    # Check if split data contains graphs directly or indices
    if 'train' in split_data and 'test' in split_data:
        # Data contains graphs directly
        all_train_data = split_data['train']
        test_data = split_data['test']
        print(f"✓ Train samples: {len(all_train_data)}")
        print(f"✓ Test samples: {len(test_data)}")
        
        # Create validation split from training data (20% of training)
        val_size = int(0.2 * len(all_train_data))
        val_data = all_train_data[:val_size]
        train_data = all_train_data[val_size:]
        print(f"✓ Validation samples: {len(val_data)} (from training set)")
        print(f"✓ Final training samples: {len(train_data)}")
    else:
        raise ValueError("Split data format not recognized. Expected 'train' and 'test' keys.")
    
    # Load dataset info
    info_path = os.path.join(data_dir_abs, 'dataset_info.json')
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    print(f"\nDataset Statistics:")
    print(f"  Average tasks per graph: {dataset_info['sample_statistics']['avg_num_tasks']:.1f}")
    print(f"  Average edges per graph: {dataset_info['sample_statistics']['avg_num_edges']:.1f}")
    print(f"  Min tasks: {dataset_info['sample_statistics']['min_tasks']}")
    print(f"  Max tasks: {dataset_info['sample_statistics']['max_tasks']}")
    print(f"  Number of classes: {dataset_info['feature_dimensions']['num_classes']}")
    
    return train_data, val_data, test_data, dataset_info


def create_dataloaders(train_data, val_data, test_data, batch_size=32, num_workers=0):
    """
    Create PyTorch Geometric data loaders
    
    Args:
        train_data: Training graph list
        val_data: Validation graph list
        test_data: Test graph list
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print(f"\nCreating data loaders (batch_size={batch_size})...")
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def save_results(metrics, output_dir='models'):
    """
    Save evaluation results
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save results
    """
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Test Set Evaluation Results")
    print(f"{'='*70}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    tier_names = ['Edge', 'Fog', 'Cloud']
    num_classes_found = len(metrics['per_class_metrics']['precision'])
    
    for i in range(num_classes_found):
        tier = tier_names[i] if i < len(tier_names) else f"Class {i}"
        print(f"  {tier}:")
        print(f"    Precision: {metrics['per_class_metrics']['precision'][i]:.4f}")
        print(f"    Recall:    {metrics['per_class_metrics']['recall'][i]:.4f}")
        print(f"    F1-Score:  {metrics['per_class_metrics']['f1_score'][i]:.4f}")
        print(f"    Support:   {metrics['per_class_metrics']['support'][i]}")
    
    # Show which classes were not found
    if num_classes_found < len(tier_names):
        print(f"\n  Note: Only {num_classes_found} class(es) found in test data.")
        missing_classes = tier_names[num_classes_found:]
        print(f"  Missing: {', '.join(missing_classes)}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    cm_size = len(cm)
    
    # Print header
    print(f"              Predicted")
    header = "              "
    for i in range(cm_size):
        header += f"{tier_names[i] if i < len(tier_names) else f'C{i}':5s} "
    print(header)
    
    # Print rows
    for i in range(cm_size):
        tier = tier_names[i] if i < len(tier_names) else f"C{i}"
        row = f"Actual {tier:5s}  "
        for j in range(cm_size):
            row += f"{cm[i][j]:4d} "
        print(row)
    
    print(f"\nResults saved to {results_path}")


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train GNN for Task-to-Resource Mapping')
    parser.add_argument('--data_dir', type=str, default='gnn_solution',
                        help='Directory containing preprocessed data')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models and results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads in GAT')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print(f"GNN Training for Task-to-Resource Mapping")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load dataset
    train_data, val_data, test_data, dataset_info = load_dataset(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"\n{'='*70}")
    print("Creating GNN model...")
    print(f"{'='*70}")
    
    model_config = {
        'node_feature_dim': dataset_info['feature_dimensions']['node_feature_dim'],
        'edge_feature_dim': dataset_info['feature_dimensions']['edge_feature_dim'],
        'platform_feature_dim': dataset_info['feature_dimensions']['platform_feature_dim'],
        'hidden_dim': args.hidden_dim,
        'num_classes': dataset_info['feature_dimensions']['num_classes'],
        'num_heads': args.num_heads,
        'dropout': args.dropout
    }
    
    model = create_model(model_config)
    print(f"✓ Model created with {count_parameters(model):,} parameters")
    print(f"\nModel Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Save model config
    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"\n✓ Model configuration saved to {config_path}")
    
    # Create trainer
    trainer = GNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping,
        verbose=True
    )
    
    # Load best model for evaluation
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['best_epoch']}")
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("Evaluating on test set...")
    print(f"{'='*70}")
    
    metrics, predictions, labels, probabilities = evaluate_model(
        model, test_loader, device=device
    )
    
    # Save results
    save_results(metrics, args.save_dir)
    
    # Save predictions
    predictions_data = {
        'predictions': predictions,
        'labels': labels,
        'probabilities': probabilities
    }
    predictions_path = os.path.join(args.save_dir, 'test_predictions.pt')
    torch.save(predictions_data, predictions_path)
    print(f"✓ Predictions saved to {predictions_path}")
    
    print(f"\n{'='*70}")
    print("Training completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
