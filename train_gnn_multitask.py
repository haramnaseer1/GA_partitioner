"""
Multi-Task GNN Model for Task Scheduling
==========================================

Graph Neural Network with multiple output heads:
1. Node-level: Processor assignment (classification)
2. Node-level: Start time prediction (regression)
3. Node-level: End time prediction (regression)
4. Graph-level: Makespan prediction (regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool


class MultiTaskSchedulingGNN(nn.Module):
    """
    Multi-Task Graph Attention Network for Task Scheduling
    
    Architecture:
    - Shared GATv2 backbone for feature extraction
    - 4 task-specific heads:
      1. Processor assignment (node-level classification)
      2. Start time (node-level regression)
      3. End time (node-level regression)
      4. Makespan (graph-level regression)
    
    Args:
        node_feature_dim (int): Node feature dimension (default: 3)
        edge_feature_dim (int): Edge feature dimension (default: 1)
        hidden_dim (int): Hidden layer dimension (default: 256)
        num_gat_layers (int): Number of GAT layers (default: 4)
        num_heads (int): Number of attention heads (default: 8)
        num_processors (int): Number of processors for classification (default: 192)
        dropout (float): Dropout rate (default: 0.2)
    """
    
    def __init__(
        self,
        node_feature_dim=3,
        edge_feature_dim=1,
        hidden_dim=256,
        num_gat_layers=4,
        num_heads=8,
        num_processors=192,
        dropout=0.2
    ):
        super(MultiTaskSchedulingGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.num_processors = num_processors
        self.dropout = dropout
        
        # ================================================================
        # SHARED ENCODER
        # ================================================================
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim // 4),
            nn.ReLU()
        )
        edge_dim = hidden_dim // 4
        
        # ================================================================
        # SHARED GAT BACKBONE
        # ================================================================
        
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_gat_layers):
            in_channels = hidden_dim
            
            if i < num_gat_layers - 1:
                # Multi-head with concat
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
                # Last layer: single head
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
        
        # ================================================================
        # TASK-SPECIFIC HEADS
        # ================================================================
        
        # HEAD 1: Processor Assignment (Node-level Classification)
        self.processor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(hidden_dim // 2, num_processors)  # Output: [num_nodes, 192]
        )
        
        # HEAD 2: Start Time (Node-level Regression)
        self.start_time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 4, 1)  # Output: [num_nodes, 1]
        )
        
        # HEAD 3: End Time (Node-level Regression)
        self.end_time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 4, 1)  # Output: [num_nodes, 1]
        )
        
        # HEAD 4: Makespan (Graph-level Regression)
        # Pooling weights for combining mean/max/sum
        self.pool_weight = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        
        self.makespan_head = nn.Sequential(
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
            
            nn.Linear(hidden_dim // 2, 1)  # Output: [batch_size, 1]
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
        Forward pass - returns all task predictions
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
        
        Returns:
            dict: {
                'processor': [num_nodes, 192] - processor logits
                'start_time': [num_nodes, 1] - start time predictions
                'end_time': [num_nodes, 1] - end time predictions
                'makespan': [batch_size, 1] - makespan predictions
            }
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # ================================================================
        # SHARED FEATURE EXTRACTION
        # ================================================================
        
        # Encode node features
        x = self.node_encoder(x)
        
        # Encode edge features
        edge_attr = self.edge_encoder(data.edge_attr) if data.edge_attr is not None else None
        
        # Apply GAT layers with residual connections
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_new = gat(x, edge_index, edge_attr=edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection
            if x.size(-1) == x_new.size(-1) and i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # ================================================================
        # TASK-SPECIFIC PREDICTIONS
        # ================================================================
        
        # Node-level predictions (per task)
        processor_logits = self.processor_head(x)  # [num_nodes, 192]
        start_time_pred = self.start_time_head(x)  # [num_nodes, 1]
        end_time_pred = self.end_time_head(x)      # [num_nodes, 1]
        
        # Graph-level prediction (per graph)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        weights = F.softmax(self.pool_weight, dim=0)
        x_pool = torch.cat([
            x_mean * weights[0],
            x_max * weights[1],
            x_sum * weights[2]
        ], dim=1)
        
        makespan_pred = self.makespan_head(x_pool)  # [batch_size, 1]
        
        return {
            'processor': processor_logits,
            'start_time': start_time_pred,
            'end_time': end_time_pred,
            'makespan': makespan_pred
        }


def create_multitask_model(
    node_feature_dim=3,
    edge_feature_dim=1,
    hidden_dim=256,
    num_gat_layers=4,
    num_heads=8,
    num_processors=192,
    dropout=0.2
):
    """
    Create multi-task GNN model
    
    Returns:
        MultiTaskSchedulingGNN: The model instance
    """
    model = MultiTaskSchedulingGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        num_gat_layers=num_gat_layers,
        num_heads=num_heads,
        num_processors=num_processors,
        dropout=dropout
    )
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    model = create_multitask_model()
    num_params = count_parameters(model)
    
    print(f"Multi-Task Scheduling GNN")
    print(f"  Total parameters: {num_params:,}")
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    from torch_geometric.data import Data
    
    # Create dummy data (5 nodes, 4 edges)
    x = torch.randn(5, 3)  # [num_nodes, 3] - node features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 1)  # [num_edges, 1]
    batch = torch.zeros(5, dtype=torch.long)  # All nodes in same graph
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(data)
    
    print(f"\n{'='*70}")
    print("Forward Pass Test:")
    print(f"{'='*70}")
    print(f"Input shape: {x.shape}")
    print(f"\nOutput shapes:")
    print(f"  Processor logits: {outputs['processor'].shape}")
    print(f"  Start time: {outputs['start_time'].shape}")
    print(f"  End time: {outputs['end_time'].shape}")
    print(f"  Makespan: {outputs['makespan'].shape}")
    print(f"{'='*70}")
