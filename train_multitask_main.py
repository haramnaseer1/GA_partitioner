"""
Multi-Task GNN Training Script
================================

Train multi-task GNN model to predict:
1. Processor assignment (classification)
2. Start time (regression)
3. End time (regression)
4. Makespan (regression)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import os
import argparse
from datetime import datetime

from train_gnn_multitask import create_multitask_model, count_parameters


class MultiTaskTrainer:
    """Trainer for multi-task scheduling GNN"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cpu',
        learning_rate=0.001,
        weight_decay=1e-5,
        loss_weights=None,
        save_dir='models_multitask'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Loss weights (default: equal weighting)
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
        self.start_criterion = nn.L1Loss()  # MAE
        self.end_criterion = nn.L1Loss()    # MAE
        self.makespan_criterion = nn.L1Loss()  # MAE
        
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
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute losses
            losses = self.compute_losses(outputs, data)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            for key in metrics:
                metrics[key] += losses[key].item() if key != 'processor_acc' else losses[key].item()
            num_batches += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def validate(self):
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
            for data in self.val_loader:
                data = data.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute losses
                losses = self.compute_losses(outputs, data)
                
                # Accumulate metrics
                for key in metrics:
                    metrics[key] += losses[key].item() if key != 'processor_acc' else losses[key].item()
                num_batches += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def train(self, num_epochs):
        """Train the model for multiple epochs"""
        print(f"\n{'='*70}")
        print(f"MULTI-TASK GNN TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Loss weights: {self.loss_weights}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
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
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train >> Total={train_metrics['total']:.4f}, "
                  f"Proc={train_metrics['processor']:.4f} (Acc={train_metrics['processor_acc']*100:.1f}%), "
                  f"Start={train_metrics['start']:.4f}, "
                  f"End={train_metrics['end']:.4f}, "
                  f"Makespan={train_metrics['makespan']:.4f}")
            print(f"  Val   >> Total={val_metrics['total']:.4f}, "
                  f"Proc={val_metrics['processor']:.4f} (Acc={val_metrics['processor_acc']*100:.1f}%), "
                  f"Start={val_metrics['start']:.4f}, "
                  f"End={val_metrics['end']:.4f}, "
                  f"Makespan={val_metrics['makespan']:.4f}")
            
            # Save best model
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.best_epoch = epoch + 1
                self.save_checkpoint('best_model.pt')
                print(f"  [BEST MODEL SAVED] Val Loss: {self.best_val_loss:.4f}")
            
            print()
        
        print(f"{'='*70}")
        print(f"Training Complete!")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Task Scheduling GNN')
    parser.add_argument('--data', type=str, default='training_data_multitask.pt',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of GAT layers')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data_list = torch.load(args.data, weights_only=False)
    print(f"Loaded {len(data_list)} graphs")
    
    if len(data_list) == 0:
        print("ERROR: No data found!")
        return
    
    # Check first graph structure
    sample = data_list[0]
    print(f"\nData structure:")
    print(f"  Node features: {sample.x.shape}")
    print(f"  Edge index: {sample.edge_index.shape}")
    print(f"  Edge attr: {sample.edge_attr.shape if hasattr(sample, 'edge_attr') else 'None'}")
    print(f"  y_processor: {sample.y_processor.shape}")
    print(f"  y_start: {sample.y_start.shape}")
    print(f"  y_end: {sample.y_end.shape}")
    print(f"  y_makespan: {sample.y_makespan.shape}")
    
    # Split data (80/20 train/val)
    if len(data_list) > 1:
        train_size = int(0.8 * len(data_list))
        train_data = data_list[:train_size]
        val_data = data_list[train_size:]
    else:
        # For testing with single graph
        print("\nWARNING: Only 1 graph available - using same for train/val")
        train_data = data_list
        val_data = data_list
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} graphs")
    print(f"  Val: {len(val_data)} graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"\nCreating model...")
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
    print(f"Model created with {num_params:,} parameters")
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
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
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    print(f"Models saved to: {trainer.save_dir}/")


if __name__ == '__main__':
    main()
