"""
GNN Training Pipeline
=====================

Training script for the Task-to-Resource Mapping GNN model.
Handles data loading, model training, validation, and checkpointing.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from datetime import datetime

from src.gnn_model import TaskResourceGNN, create_model, count_parameters


class GNNTrainer:
    """
    Trainer class for GNN model training and evaluation
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cpu',
        learning_rate=0.001,
        weight_decay=1e-4,
        save_dir='models'
    ):
        """
        Initialize trainer
        
        Args:
            model: GNN model instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization parameter
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Loss function (CrossEntropyLoss for multi-class classification)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (Adam with weight decay)
        self.optimizer = optim.Adam(
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
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(data)
            
            # Compute loss
            loss = self.criterion(out, data.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        Validate the model
        
        Returns:
            tuple: (average_loss, accuracy, precision, recall, f1, confusion_matrix)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                
                # Forward pass
                out = self.model(data)
                
                # Compute loss
                loss = self.criterion(out, data.y)
                total_loss += loss.item()
                
                # Track predictions
                preds = torch.argmax(out, dim=1).cpu().numpy()
                labels = data.y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Compute additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        return avg_loss, accuracy, precision, recall, f1, cm
    
    def train(self, num_epochs, early_stopping_patience=20, verbose=True):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
            verbose: Print training progress
            
        Returns:
            dict: Training history
        """
        print(f"\n{'='*70}")
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, cm = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"F1: {val_f1:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, 'best_model.pt')
                patience_counter = 0
                
                if verbose:
                    print(f"  → New best model! Accuracy: {val_acc:.4f}")
                    print(f"  → Confusion Matrix:\n{cm}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                print(f"  Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print(f"{'='*70}\n")
        
        # Save final checkpoint
        self.save_checkpoint(epoch, 'final_model.pt')
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, filename):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def save_history(self):
        """
        Save training history to JSON
        """
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resumed from epoch {checkpoint['epoch']}")


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set
    
    Args:
        model: Trained GNN model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support.tolist()
        }
    }
    
    return metrics, all_preds, all_labels, all_probs


if __name__ == "__main__":
    print("GNN Training Pipeline - Test")
    print("This module should be imported and used by main training script")
