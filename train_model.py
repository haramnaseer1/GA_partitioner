"""
Multi-Task GNN Training Script
================================

Train multi-task scheduling GNN with processor assignment, timing, and makespan prediction.

Usage:
    python train_model.py                              # Default: 50 epochs
    python train_model.py --epochs 100                 # Custom epochs
    python train_model.py --quick                      # Quick test: 10 epochs
    python train_model.py --device cuda                # Use GPU
    python train_model.py --resume models_multitask/best_model.pt  # Resume training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import os
import argparse
from datetime import datetime
import json

from train_gnn_multitask import create_multitask_model, count_parameters


class MultiTaskTrainer:
    """Trainer for multi-task scheduling GNN"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device='cpu',
        learning_rate=0.001,
        weight_decay=1e-5,
        loss_weights=None,
        save_dir='models_multitask'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        # Loss weights
        if loss_weights is None:
            loss_weights = {
                'processor': 1.0,
                'start_time': 1.0,
                'end_time': 1.0,
                'makespan': 1.0
            }
        self.loss_weights = loss_weights
        
        # Loss functions
        self.processor_criterion = nn.CrossEntropyLoss()
        self.start_criterion = nn.L1Loss()
        self.end_criterion = nn.L1Loss()
        self.makespan_criterion = nn.L1Loss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_total_loss': [],
            'train_processor_loss': [],
            'train_processor_acc': [],
            'train_start_loss': [],
            'train_end_loss': [],
            'train_makespan_loss': [],
            'val_total_loss': [],
            'val_processor_loss': [],
            'val_processor_acc': [],
            'val_start_loss': [],
            'val_end_loss': [],
            'val_makespan_loss': [],
            'learning_rate': []
        }
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.start_epoch = 0
    
    def compute_losses(self, outputs, data):
        """Compute all task losses"""
        # Processor assignment (classification)
        processor_loss = self.processor_criterion(
            outputs['processor'],
            data.y_processor
        )
        
        # Processor accuracy
        processor_pred = outputs['processor'].argmax(dim=1)
        processor_acc = (processor_pred == data.y_processor).float().mean()
        
        # Start time (regression)
        start_loss = self.start_criterion(
            outputs['start_time'].squeeze(),
            data.y_start
        )
        
        # End time (regression)
        end_loss = self.end_criterion(
            outputs['end_time'].squeeze(),
            data.y_end
        )
        
        # Makespan (graph-level regression)
        makespan_loss = self.makespan_criterion(
            outputs['makespan'].squeeze(),
            data.y_makespan
        )
        
        # Total weighted loss
        total_loss = (
            self.loss_weights['processor'] * processor_loss +
            self.loss_weights['start_time'] * start_loss +
            self.loss_weights['end_time'] * end_loss +
            self.loss_weights['makespan'] * makespan_loss
        )
        
        return {
            'total': total_loss,
            'processor': processor_loss,
            'processor_acc': processor_acc,
            'start': start_loss,
            'end': end_loss,
            'makespan': makespan_loss
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        metrics = {
            'total': 0.0,
            'processor': 0.0,
            'processor_acc': 0.0,
            'start': 0.0,
            'end': 0.0,
            'makespan': 0.0
        }
        num_batches = 0
        
        for data in self.train_loader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            losses = self.compute_losses(outputs, data)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key in metrics:
                metrics[key] += losses[key].item() if key != 'processor_acc' else losses[key].item()
            num_batches += 1
        
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def validate(self, loader):
        """Validate the model"""
        self.model.eval()
        
        metrics = {
            'total': 0.0,
            'processor': 0.0,
            'processor_acc': 0.0,
            'start': 0.0,
            'end': 0.0,
            'makespan': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                outputs = self.model(data)
                losses = self.compute_losses(outputs, data)
                
                for key in metrics:
                    metrics[key] += losses[key].item() if key != 'processor_acc' else losses[key].item()
                num_batches += 1
        
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def train(self, num_epochs):
        """Train the model"""
        print(f"\n{'='*70}")
        print(f"MULTI-TASK GNN TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {self.start_epoch + 1} -> {self.start_epoch + num_epochs}")
        print(f"Device: {self.device}")
        print(f"Loss weights: {self.loss_weights}")
        print(f"Save directory: {self.save_dir}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            current_epoch = self.start_epoch + epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['total'])
            
            # Record history
            self.history['train_total_loss'].append(train_metrics['total'])
            self.history['train_processor_loss'].append(train_metrics['processor'])
            self.history['train_processor_acc'].append(train_metrics['processor_acc'])
            self.history['train_start_loss'].append(train_metrics['start'])
            self.history['train_end_loss'].append(train_metrics['end'])
            self.history['train_makespan_loss'].append(train_metrics['makespan'])
            
            self.history['val_total_loss'].append(val_metrics['total'])
            self.history['val_processor_loss'].append(val_metrics['processor'])
            self.history['val_processor_acc'].append(val_metrics['processor_acc'])
            self.history['val_start_loss'].append(val_metrics['start'])
            self.history['val_end_loss'].append(val_metrics['end'])
            self.history['val_makespan_loss'].append(val_metrics['makespan'])
            
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"Epoch {current_epoch}/{self.start_epoch + num_epochs}")
            print(f"  Train >> Loss={train_metrics['total']:.2f}, "
                  f"Proc={train_metrics['processor']:.2f} ({train_metrics['processor_acc']*100:.1f}%), "
                  f"Start={train_metrics['start']:.2f}, "
                  f"End={train_metrics['end']:.2f}, "
                  f"Mksp={train_metrics['makespan']:.2f}")
            print(f"  Val   >> Loss={val_metrics['total']:.2f}, "
                  f"Proc={val_metrics['processor']:.2f} ({val_metrics['processor_acc']*100:.1f}%), "
                  f"Start={val_metrics['start']:.2f}, "
                  f"End={val_metrics['end']:.2f}, "
                  f"Mksp={val_metrics['makespan']:.2f}")
            
            # Save best model
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.best_epoch = current_epoch
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ BEST MODEL (Val Loss: {self.best_val_loss:.4f})")
            
            print()
        
        # Final evaluation on test set
        if self.test_loader:
            print(f"{'='*70}")
            print("FINAL TEST EVALUATION")
            print(f"{'='*70}")
            test_metrics = self.validate(self.test_loader)
            print(f"Test >> Loss={test_metrics['total']:.2f}, "
                  f"Proc={test_metrics['processor']:.2f} ({test_metrics['processor_acc']*100:.1f}%), "
                  f"Start={test_metrics['start']:.2f}, "
                  f"End={test_metrics['end']:.2f}, "
                  f"Mksp={test_metrics['makespan']:.2f}")
            print(f"{'='*70}\n")
        
        print(f"Training Complete!")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Models saved to: {self.save_dir}/\n")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.start_epoch + len(self.history['train_total_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {self.start_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Multi-Task Scheduling GNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                              # Default training
  python train_model.py --epochs 100 --device cuda   # 100 epochs on GPU
  python train_model.py --quick                      # Quick test (10 epochs)
  python train_model.py --resume best_model.pt       # Resume from checkpoint
        """
    )
    
    parser.add_argument('--data', type=str, default='training_data_multitask.pt',
                        help='Path to training data (default: training_data_multitask.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of GAT layers (default: 4)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (10 epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Training data fraction (default: 0.7)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation data fraction (default: 0.15)')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.epochs = 10
        print("\n*** QUICK TEST MODE: 10 epochs ***\n")
    
    # Load data
    print(f"\n{'='*70}")
    print(f"LOADING DATA")
    print(f"{'='*70}")
    print(f"Data file: {args.data}")
    
    if not os.path.exists(args.data):
        print(f"\nERROR: Data file not found: {args.data}")
        print("Please run: python generate_training_data.py")
        return 1
    
    data_list = torch.load(args.data, weights_only=False)
    print(f"Loaded {len(data_list)} graphs")
    
    if len(data_list) == 0:
        print("\nERROR: No training data found!")
        return 1
    
    # Data info
    sample = data_list[0]
    print(f"\nData structure:")
    print(f"  Node features: {sample.x.shape}")
    print(f"  Edge index: {sample.edge_index.shape}")
    print(f"  Targets: processor, start_time, end_time, makespan")
    
    # Split data
    n = len(data_list)
    train_size = int(args.train_split * n)
    val_size = int(args.val_split * n)
    test_size = n - train_size - val_size
    
    # Ensure at least 1 sample in each split
    if test_size < 1 and n > 2:
        test_size = 1
        val_size = min(val_size, n - train_size - 1)
        train_size = n - val_size - test_size
    
    train_data = data_list[:train_size]
    val_data = data_list[train_size:train_size+val_size]
    test_data = data_list[train_size+val_size:] if test_size > 0 else None
    
    print(f"\nDataset split ({n} total):")
    print(f"  Train: {len(train_data)} ({100*len(train_data)/n:.1f}%)")
    print(f"  Val:   {len(val_data)} ({100*len(val_data)/n:.1f}%)")
    if test_data:
        print(f"  Test:  {len(test_data)} ({100*len(test_data)/n:.1f}%)")
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False) if test_data else None
    
    # Create model
    print(f"\n{'='*70}")
    print(f"MODEL CREATION")
    print(f"{'='*70}")
    
    model = create_multitask_model(
        node_feature_dim=3,
        edge_feature_dim=1,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_layers,
        num_heads=args.num_heads,
        num_processors=192,
        dropout=args.dropout
    )
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    print(f"Device: {args.device}")
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=1e-5,
        loss_weights={
            'processor': 1.0,
            'start_time': 1.0,
            'end_time': 1.0,
            'makespan': 1.0
        }
    )
    
    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    print(f"✓ Training complete!")
    print(f"  Best model: models_multitask/best_model.pt")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
