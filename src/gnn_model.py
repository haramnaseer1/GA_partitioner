"""
GNN Model for Task-to-Resource Mapping Prediction
==================================================

This module implements a Graph Neural Network for predicting optimal
resource tier assignments (Edge/Fog/Cloud) for tasks in DAG applications.

Architecture:
- Graph Attention Network (GAT) layers for learning task dependencies
- Node-level classification for resource tier prediction
- Support for heterogeneous platform features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data


class TaskResourceGNN(nn.Module):
    """
    Graph Attention Network for Task-to-Resource Mapping
    
    Predicts the optimal resource tier (Edge/Fog/Cloud) for each task
    in a DAG application based on task properties, dependencies, and
    platform characteristics.
    
    Args:
        node_feature_dim (int): Dimension of node features (default: 10)
        edge_feature_dim (int): Dimension of edge features (default: 2)
        platform_feature_dim (int): Dimension of platform features (default: 4)
        hidden_dim (int): Hidden layer dimension (default: 128)
        num_classes (int): Number of resource tiers (default: 3)
        num_heads (int): Number of attention heads in GAT (default: 4)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(
        self,
        node_feature_dim=10,
        edge_feature_dim=2,
        platform_feature_dim=4,
        hidden_dim=128,
        num_classes=3,
        num_heads=4,
        dropout=0.3
    ):
        super(TaskResourceGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.platform_feature_dim = platform_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input projection layer
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # Platform feature encoder
        self.platform_encoder = nn.Sequential(
            nn.Linear(platform_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers with multi-head attention
        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.gat3 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Output classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, data):
        """
        Forward pass of the GNN model
        
        Args:
            data (torch_geometric.data.Data): Graph data object containing:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feature_dim]
                - platform_features: Platform characteristics [platform_feature_dim]
                - batch: Batch assignment vector (for batched graphs)
        
        Returns:
            torch.Tensor: Class logits for each node [num_nodes, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode node features
        x = self.node_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT Layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT Layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT Layer 3
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Encode platform features and broadcast to all nodes
        if hasattr(data, 'platform_x') and data.platform_x is not None:
            # data.platform_x shape: [num_platforms, platform_feature_dim]
            # For batched graphs, we need to map each node to its platform
            platform_emb = self.platform_encoder(data.platform_x)  # [num_platforms, hidden_dim//2]
            
            # Expand platform embedding to match each node in the batch
            # batch tensor tells us which graph each node belongs to
            num_nodes = x.size(0)
            
            # Create expanded platform features for each node
            # Each node gets the platform features from its graph
            if hasattr(data, 'batch') and data.batch is not None:
                # For each node, get its batch index and use corresponding platform features
                # Since each graph has same platform (index 0 for all), we just use first platform
                platform_emb_per_graph = platform_emb[0:1]  # Take first platform
                platform_emb_expanded = platform_emb_per_graph.repeat(num_nodes, 1)
            else:
                # Single graph case
                platform_emb_expanded = platform_emb.repeat(num_nodes, 1)
            
            # Concatenate node and platform features
            x = torch.cat([x, platform_emb_expanded], dim=1)
        
        # Classification
        out = self.classifier(x)
        
        return out
    
    def predict(self, data):
        """
        Predict resource tier for each task
        
        Args:
            data: Graph data object
            
        Returns:
            torch.Tensor: Predicted class labels [num_nodes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def get_attention_weights(self, data):
        """
        Extract attention weights for visualization
        
        Args:
            data: Graph data object
            
        Returns:
            list: Attention weights from each GAT layer
        """
        self.eval()
        attention_weights = []
        
        x, edge_index = data.x, data.edge_index
        
        # Encode node features
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # GAT Layer 1 with attention
        x, (edge_index_1, alpha_1) = self.gat1(x, edge_index, return_attention_weights=True)
        attention_weights.append((edge_index_1, alpha_1))
        x = self.bn1(x)
        x = F.relu(x)
        
        # GAT Layer 2 with attention
        x, (edge_index_2, alpha_2) = self.gat2(x, edge_index, return_attention_weights=True)
        attention_weights.append((edge_index_2, alpha_2))
        x = self.bn2(x)
        x = F.relu(x)
        
        # GAT Layer 3 with attention
        x, (edge_index_3, alpha_3) = self.gat3(x, edge_index, return_attention_weights=True)
        attention_weights.append((edge_index_3, alpha_3))
        
        return attention_weights


def create_model(config=None):
    """
    Factory function to create a GNN model with specified configuration
    
    Args:
        config (dict, optional): Model configuration dictionary
        
    Returns:
        TaskResourceGNN: Instantiated model
    """
    if config is None:
        config = {
            'node_feature_dim': 10,
            'edge_feature_dim': 2,
            'platform_feature_dim': 4,
            'hidden_dim': 128,
            'num_classes': 3,
            'num_heads': 4,
            'dropout': 0.3
        }
    
    model = TaskResourceGNN(**config)
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Creating TaskResourceGNN model...")
    model = create_model()
    print(f"Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"\nModel architecture:\n{model}")
