"""
Training Script for Scheduling GNN
===================================

Train GNN model to predict optimal makespan for task scheduling.
Uses training_data.pt generated from GA solutions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

from train_gnn_scheduling import create_model, count_parameters


class MakespanTrainer:
    """Trainer for makespan prediction GNN"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cpu',
        learning_rate=0.001,
        weight_decay=1e-5,
        save_dir='models_scheduling'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        
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
            patience=15,
            min_lr=1e-6,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'train_mape': [],
            'val_loss': [],
            'val_mae': [],
            'val_mape': [],
            'learning_rate': []
        }
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_mape = 0
        num_batches = 0
        
        for data in self.train_loader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(data)
            target = data.y.view(-1, 1)
            
            # Compute losses
            loss = self.criterion(pred, target)
            mae = self.mae_criterion(pred, target)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = torch.mean(torch.abs((target - pred) / (target + 1e-8))) * 100
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_mape += mape.item()
            num_batches += 1
        
        return (
            total_loss / num_batches,
            total_mae / num_batches,
            total_mape / num_batches
        )
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_mape = 0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                
                pred = self.model(data)
                target = data.y.view(-1, 1)
                
                loss = self.criterion(pred, target)
                mae = self.mae_criterion(pred, target)
                mape = torch.mean(torch.abs((target - pred) / (target + 1e-8))) * 100
                
                total_loss += loss.item()
                total_mae += mae.item()
                total_mape += mape.item()
                num_batches += 1
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        return (
            total_loss / num_batches,
            total_mae / num_batches,
            total_mape / num_batches,
            np.array(all_preds),
            np.array(all_targets)
        )
    
    def train(self, num_epochs, early_stopping_patience=30, verbose=True):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            verbose: Print progress
        """
        print(f"\n{'='*80}")
        print(f"Training Scheduling GNN on {self.device}")
        print(f"{'='*80}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"{'='*80}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_mae, train_mape = self.train_epoch()
            
            # Validate
            val_loss, val_mae, val_mape, preds, targets = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            self.history['train_mape'].append(train_mape)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_mape'].append(val_mape)
            self.history['learning_rate'].append(current_lr)
            
            if verbose:
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train: Loss={train_loss:.4f} MAE={train_mae:.2f} MAPE={train_mape:.2f}% | "
                      f"Val: Loss={val_loss:.4f} MAE={val_mae:.2f} MAPE={val_mape:.2f}% | "
                      f"LR={current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch, 'best_model.pt')
                patience_counter = 0
                
                if verbose:
                    print(f"  >> New best model! Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                print(f"  Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                break
            
            # Save periodic checkpoints
            if epoch % 20 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')
        
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print(f"{'='*80}\n")
        
        # Save final model and history
        self.save_checkpoint(epoch, 'final_model.pt')
        self.save_history()
        self.plot_training_curves()
        
        return self.history
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'model_config': {
                'node_feature_dim': self.model.node_feature_dim,
                'edge_feature_dim': self.model.edge_feature_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_gat_layers': self.model.num_gat_layers,
                'num_heads': self.model.num_heads,
                'dropout': self.model.dropout
            }
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(self.history['train_mae'], label='Train MAE', alpha=0.8)
        axes[0, 1].plot(self.history['val_mae'], label='Val MAE', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_title('MAE Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAPE
        axes[1, 0].plot(self.history['train_mape'], label='Train MAPE', alpha=0.8)
        axes[1, 0].plot(self.history['val_mape'], label='Val MAPE', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].set_title('Mean Absolute Percentage Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['learning_rate'], alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved to {plot_path}")
        plt.close()


def prepare_data(data_path='training_data.pt', batch_size=32, val_split=0.15, test_split=0.15):
    """
    Load and prepare data for training
    
    Args:
        data_path: Path to training_data.pt
        batch_size: Batch size for DataLoader
        val_split: Validation split ratio
        test_split: Test split ratio
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print(f"\nLoading data from {data_path}...")
    
    # Load data
    data_list = torch.load(data_path, weights_only=False)
    print(f"Loaded {len(data_list)} graphs")
    
    # Convert to PyG Data objects
    pyg_data_list = []
    for d in data_list:
        pyg_data = Data(
            x=d['x'],
            edge_index=d['edge_index'],
            edge_attr=d['edge_attr'],
            y=d['y']
        )
        pyg_data_list.append(pyg_data)
    
    # Split data
    indices = list(range(len(pyg_data_list)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_split/(1-test_split), random_state=42
    )
    
    train_data = [pyg_data_list[i] for i in train_idx]
    val_data = [pyg_data_list[i] for i in val_idx]
    test_data = [pyg_data_list[i] for i in test_idx]
    
    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'node_feature_dim': 6,
        'edge_feature_dim': 1,
        'hidden_dim': 256,
        'num_gat_layers': 4,
        'num_heads': 8,
        'dropout': 0.2
    }
    
    training_config = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 200,
        'early_stopping_patience': 30,
        'val_split': 0.15,
        'test_split': 0.15
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = prepare_data(
        data_path='training_data.pt',
        batch_size=training_config['batch_size'],
        val_split=training_config['val_split'],
        test_split=training_config['test_split']
    )
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(config, model_type='standard')
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = MakespanTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        save_dir='models_scheduling'
    )
    
    # Train model
    history = trainer.train(
        num_epochs=training_config['num_epochs'],
        early_stopping_patience=training_config['early_stopping_patience'],
        verbose=True
    )
    
    print("\n✓ Training completed successfully!")
    print(f"Models saved in: models_scheduling/")
    

if __name__ == "__main__":
    main()
