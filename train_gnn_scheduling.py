"""
Improved GNN Model for Task Scheduling Makespan Prediction
============================================================

Graph Neural Network to predict optimal scheduling makespan.
Uses node features (task properties), edge features (messages),
and graph-level prediction for makespan optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool


class SchedulingGNN(nn.Module):
    """
    Graph Attention Network for Task Scheduling Makespan Prediction
    
    Architecture:
    - GATv2 layers for learning task dependencies with attention
    - Edge features (message sizes) incorporated
    - Graph-level pooling for makespan prediction
    - Multi-head attention for capturing different dependency patterns
    
    Args:
        node_feature_dim (int): Node feature dimension (default: 6)
        edge_feature_dim (int): Edge feature dimension (default: 1)
        hidden_dim (int): Hidden layer dimension (default: 256)
        num_gat_layers (int): Number of GAT layers (default: 4)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout rate (default: 0.2)
        use_edge_features (bool): Whether to use edge features (default: True)
    """
    
    def __init__(
        self,
        node_feature_dim=6,
        edge_feature_dim=1,
        hidden_dim=256,
        num_gat_layers=4,
        num_heads=8,
        dropout=0.2,
        use_edge_features=True
    ):
        super(SchedulingGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature encoder (if using edge features)
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim // 4),
                nn.ReLU()
            )
            edge_dim = hidden_dim // 4
        else:
            edge_dim = None
        
        # GAT layers with multi-head attention
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_gat_layers):
            if i == 0:
                in_channels = hidden_dim
            else:
                in_channels = hidden_dim
            
            # Use concat=True for all layers except last
            if i < num_gat_layers - 1:
                self.gat_layers.append(
                    GATv2Conv(
                        in_channels,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        concat=True
                    )
                )
            else:
                # Last layer: single head, no concat
                self.gat_layers.append(
                    GATv2Conv(
                        in_channels,
                        hidden_dim,
                        heads=1,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        concat=False
                    )
                )
            
            self.batch_norms.append(nn.LayerNorm(hidden_dim))
        
        # Graph-level readout (combine different pooling strategies)
        self.pool_weight = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        
        # Regressor head for makespan prediction
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
        
        Returns:
            torch.Tensor: Predicted makespan [batch_size, 1]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode node features
        x = self.node_encoder(x)
        
        # Encode edge features if available
        if self.use_edge_features and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = None
        
        # Apply GAT layers
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_new = gat(x, edge_index, edge_attr=edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if x.size(-1) == x_new.size(-1) and i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # Graph-level pooling (combine mean, max, and sum)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        # Weighted combination of pooling strategies
        weights = F.softmax(self.pool_weight, dim=0)
        x_pool = torch.cat([
            x_mean * weights[0],
            x_max * weights[1],
            x_sum * weights[2]
        ], dim=1)
        
        # Predict makespan
        out = self.regressor(x_pool)
        
        return out
    
    def get_attention_weights(self, data):
        """
        Extract attention weights for visualization
        
        Args:
            data: PyG Data object
        
        Returns:
            list: Attention weights from each GAT layer
        """
        self.eval()
        attention_weights = []
        
        x, edge_index = data.x, data.edge_index
        
        # Encode features
        x = self.node_encoder(x)
        
        if self.use_edge_features and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = None
        
        # Extract attention from each layer
        with torch.no_grad():
            for gat in self.gat_layers:
                x, (edge_idx, alpha) = gat(
                    x, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True
                )
                attention_weights.append((edge_idx, alpha))
                x = F.relu(x)
        
        return attention_weights


class SchedulingGNNDeep(nn.Module):
    """
    Deeper variant with more layers and residual connections
    """
    
    def __init__(
        self,
        node_feature_dim=6,
        edge_feature_dim=1,
        hidden_dim=512,
        num_gat_layers=6,
        num_heads=8,
        dropout=0.3
    ):
        super(SchedulingGNNDeep, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Deeper GAT stack with residual connections
        self.gat_blocks = nn.ModuleList()
        for _ in range(num_gat_layers):
            self.gat_blocks.append(
                nn.ModuleDict({
                    'gat': GATv2Conv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=hidden_dim // 4,
                        concat=True
                    ),
                    'norm': nn.LayerNorm(hidden_dim),
                    'ffn': nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 2, hidden_dim)
                    ),
                    'norm2': nn.LayerNorm(hidden_dim)
                })
            )
        
        # Graph pooling and regression
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x = self.encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr) if data.edge_attr is not None else None
        
        for block in self.gat_blocks:
            # GAT with residual
            x_res = x
            x = block['gat'](x, data.edge_index, edge_attr=edge_attr)
            x = block['norm'](x)
            x = F.relu(x) + x_res
            
            # Feed-forward with residual
            x_res = x
            x = block['ffn'](x)
            x = block['norm2'](x)
            x = x + x_res
        
        # Pool and predict
        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        x_sum = global_add_pool(x, data.batch)
        x_pool = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        return self.regressor(x_pool)


def create_model(config=None, model_type='standard'):
    """
    Factory function to create GNN model
    
    Args:
        config (dict): Model configuration
        model_type (str): 'standard' or 'deep'
    
    Returns:
        nn.Module: GNN model instance
    """
    if config is None:
        config = {
            'node_feature_dim': 6,
            'edge_feature_dim': 1,
            'hidden_dim': 256,
            'num_gat_layers': 4,
            'num_heads': 8,
            'dropout': 0.2
        }
    
    if model_type == 'deep':
        model = SchedulingGNNDeep(**config)
    else:
        model = SchedulingGNN(**config)
    
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing SchedulingGNN model...")
    
    model = create_model()
    print(f"\nModel created: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    from torch_geometric.data import Data
    
    test_data = Data(
        x=torch.randn(50, 6),
        edge_index=torch.randint(0, 50, (2, 100)),
        edge_attr=torch.randn(100, 1),
        batch=torch.zeros(50, dtype=torch.long)
    )
    
    output = model(test_data)
    print(f"\nTest forward pass:")
    print(f"  Input: {test_data.num_nodes} nodes, {test_data.num_edges} edges")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.item():.2f}")
    print("\n✓ Model test passed!")
