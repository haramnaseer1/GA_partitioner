"""
Constraint-Aware GNN Model
===========================

GNN model that LEARNS to satisfy constraints during training:
1. Eligibility masking in forward pass
2. Precedence-aware loss functions
3. Non-overlap penalty losses
4. Duration consistency enforcement

Model architecture mein constraints embedded hain, not just post-processing.
"""

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
import argparse
import os


class ConstraintAwareGNN(nn.Module):
    """
    GNN model jo constraints ko follow karte hue predictions karta hai
    
    Key Features:
    - Eligibility masking during processor selection
    - Precedence-aware start time prediction
    - Duration consistency in end time calculation
    - Loss functions with constraint penalties
    """
    
    def __init__(
        self,
        node_feature_dim=3,
        edge_feature_dim=1,
        hidden_dim=128,
        num_gat_layers=2,
        num_heads=4,
        num_processors=192,
        dropout=0.2,
        platform_info=None,
        app_constraints=None
    ):
        super(ConstraintAwareGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_processors = num_processors
        self.platform_info = platform_info
        self.app_constraints = app_constraints
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim // 4),
            nn.ReLU()
        )
        edge_dim = hidden_dim // 4
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_gat_layers):
            if i < num_gat_layers - 1:
                self.gat_layers.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        concat=True
                    )
                )
            else:
                self.gat_layers.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=1,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        concat=False
                    )
                )
            self.batch_norms.append(nn.LayerNorm(hidden_dim))
        
        # ================================================================
        # CONSTRAINT-AWARE HEADS
        # ================================================================
        
        # 1. Processor Assignment with ELIGIBILITY MASKING
        self.processor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_processors)
        )
        
        # 2. Start Time Prediction (will be adjusted by precedence)
        self.start_time_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 3. Duration Prediction (processing_time / clock_speed)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim + num_processors, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 4. Communication Delay Predictor
        self.comm_delay_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable tier delay parameters (initialized to known values)
        self.tier_delays = nn.Parameter(torch.tensor([
            50.0,   # Edge-Edge
            150.0,  # Edge-Fog
            350.0,  # Edge-Cloud
            200.0,  # Fog-Cloud
            175.0   # Cloud-Cloud
        ]))
    
    def apply_eligibility_mask(self, processor_logits, can_run_on_mask):
        """
        Apply eligibility constraint: mask out invalid processors
        
        Args:
            processor_logits: [num_nodes, num_processors]
            can_run_on_mask: [num_nodes, num_processors] (1=eligible, 0=not)
        
        Returns:
            Masked logits
        """
        # Set logits of ineligible processors to -inf
        masked_logits = processor_logits.clone()
        masked_logits = masked_logits * can_run_on_mask + (-1e9) * (1 - can_run_on_mask)
        return masked_logits
    
    def compute_precedence_constraints(self, node_features, edge_index, edge_attr, 
                                      processor_probs, base_start_times):
        """
        Compute start times respecting precedence constraints
        
        Args:
            node_features: Node embeddings
            edge_index: [2, num_edges]
            edge_attr: Edge features
            processor_probs: [num_nodes, num_processors] softmax probabilities
            base_start_times: [num_nodes, 1] initial predictions
        
        Returns:
            Adjusted start times respecting dependencies
        """
        num_nodes = node_features.size(0)
        adjusted_start_times = base_start_times.clone()

        if edge_index.size(1) == 0:
            return adjusted_start_times

        # Collect all receiver indices and their new candidate start times
        receiver_indices = []
        candidate_starts = []

        for i in range(edge_index.size(1)):
            sender_idx = edge_index[0, i]
            receiver_idx = edge_index[1, i]

            edge_feat = edge_attr[i] if edge_attr is not None else torch.zeros(1)
            sender_feat = node_features[sender_idx]
            receiver_feat = node_features[receiver_idx]
            combined = torch.cat([
                sender_feat,
                receiver_feat,
                edge_feat.unsqueeze(0) if edge_feat.dim() == 0 else edge_feat
            ])
            comm_delay = F.relu(self.comm_delay_head(combined.unsqueeze(0)))
            sender_proc_dist = processor_probs[sender_idx]
            receiver_proc_dist = processor_probs[receiver_idx]
            sender_start = base_start_times[sender_idx]
            sender_duration = torch.tensor(10.0)  # Placeholder
            sender_end = sender_start + sender_duration
            min_receiver_start = sender_end + comm_delay

            receiver_indices.append(receiver_idx)
            candidate_starts.append(min_receiver_start)

        # Out-of-place update: for each receiver, set its start time to max of current and candidate
        if receiver_indices:
            receiver_indices_tensor = torch.tensor(receiver_indices, dtype=torch.long)
            candidate_starts_tensor = torch.cat(candidate_starts).view(-1, 1)
            adjusted_start_times[receiver_indices_tensor] = torch.max(
                adjusted_start_times[receiver_indices_tensor],
                candidate_starts_tensor
            )
        
        return adjusted_start_times
    
    def forward(self, data, can_run_on_masks=None, enforce_constraints=True):
        """
        Forward pass with constraint enforcement
        
        Args:
            data: PyG Data object
            can_run_on_masks: [num_nodes, num_processors] eligibility masks
            enforce_constraints: Whether to apply constraints during forward pass
        
        Returns:
            dict: Predictions with constraints applied
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(data.edge_attr) if data.edge_attr is not None else None
        
        # GAT layers
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_new = gat(x, edge_index, edge_attr=edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=0.2, training=self.training)
            
            if x.size(-1) == x_new.size(-1) and i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # ================================================================
        # 1. PROCESSOR ASSIGNMENT WITH ELIGIBILITY
        # ================================================================
        processor_logits = self.processor_head(x)
        
        if enforce_constraints and can_run_on_masks is not None:
            processor_logits = self.apply_eligibility_mask(processor_logits, can_run_on_masks)
        
        processor_probs = F.softmax(processor_logits, dim=-1)
        
        # ================================================================
        # 2. START TIME WITH PRECEDENCE
        # ================================================================
        base_start_times = F.relu(self.start_time_base(x))
        
        if enforce_constraints and edge_index.size(1) > 0:
            start_times = self.compute_precedence_constraints(
                x, edge_index, edge_attr, processor_probs, base_start_times
            )
        else:
            start_times = base_start_times
        
        # ================================================================
        # 3. DURATION (depends on processor and task)
        # ================================================================
        # Combine node features with processor assignment
        processor_assignment_features = processor_probs  # Soft assignment
        duration_input = torch.cat([x, processor_assignment_features], dim=-1)
        durations = F.relu(self.duration_head(duration_input))
        
        # ================================================================
        # 4. END TIME (start + duration)
        # ================================================================
        end_times = start_times + durations
        
        # ================================================================
        # 5. MAKESPAN (max end time)
        # ================================================================
        # Group by batch and take max
        unique_batches = torch.unique(batch)
        makespans = []
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_max = end_times[batch_mask].max()
            makespans.append(batch_max)
        makespan = torch.stack(makespans).unsqueeze(1)
        
        return {
            'processor': processor_logits,
            'processor_probs': processor_probs,
            'start_time': start_times,
            'end_time': end_times,
            'duration': durations,
            'makespan': makespan
        }


class ConstraintAwareLoss(nn.Module):
    """
    Loss function jo constraints violations ko penalize karta hai
    """
    
    def __init__(self, 
                 processor_weight=1.0,
                 start_weight=1.0,
                 end_weight=1.0,
                 makespan_weight=1.0,
                 precedence_penalty=10.0,
                 overlap_penalty=10.0,
                 duration_penalty=5.0):
        super(ConstraintAwareLoss, self).__init__()
        
        self.processor_weight = processor_weight
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.makespan_weight = makespan_weight
        self.precedence_penalty = precedence_penalty
        self.overlap_penalty = overlap_penalty
        self.duration_penalty = duration_penalty
        
        self.processor_loss_fn = nn.CrossEntropyLoss()
        self.regression_loss_fn = nn.L1Loss()
    
    def compute_precedence_violation(self, outputs, data):
        """
        Precedence constraint violation penalty
        
        For each edge u->v: start_time[v] >= end_time[u]
        """
        if data.edge_index.size(1) == 0:
            return torch.tensor(0.0, device=outputs['start_time'].device)
        
        violations = 0.0
        for i in range(data.edge_index.size(1)):
            sender = data.edge_index[0, i]
            receiver = data.edge_index[1, i]
            
            sender_end = outputs['end_time'][sender]
            receiver_start = outputs['start_time'][receiver]
            
            # Violation if receiver starts before sender ends
            violation = F.relu(sender_end - receiver_start)
            violations = violations + violation
        
        return violations / max(data.edge_index.size(1), 1)
    
    def compute_overlap_violation(self, outputs, data):
        """
        Non-overlap constraint violation penalty
        
        Tasks on same processor should not overlap
        """
        processor_assignments = outputs['processor'].argmax(dim=1)
        violations = 0.0
        count = 0
        
        # Check pairs of tasks
        num_tasks = outputs['start_time'].size(0)
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                if processor_assignments[i] == processor_assignments[j]:
                    # Same processor - check overlap
                    start_i = outputs['start_time'][i]
                    end_i = outputs['end_time'][i]
                    start_j = outputs['start_time'][j]
                    end_j = outputs['end_time'][j]
                    
                    # Overlap if: start_i < end_j AND start_j < end_i
                    overlap = F.relu(torch.min(end_i, end_j) - torch.max(start_i, start_j))
                    violations = violations + overlap
                    count += 1
        
        return violations / max(count, 1)
    
    def compute_duration_consistency(self, outputs):
        """
        Duration consistency: end_time = start_time + duration
        """
        predicted_duration = outputs['end_time'] - outputs['start_time']
        expected_duration = outputs['duration']
        
        return self.regression_loss_fn(predicted_duration, expected_duration)
    
    def forward(self, outputs, data):
        """
        Compute total loss with constraint penalties
        """
        # Basic prediction losses
        processor_loss = self.processor_loss_fn(outputs['processor'], data.y_processor)
        start_loss = self.regression_loss_fn(outputs['start_time'].squeeze(-1), data.y_start)
        end_loss = self.regression_loss_fn(outputs['end_time'].squeeze(-1), data.y_end)
        makespan_loss = self.regression_loss_fn(outputs['makespan'].squeeze(-1), data.y_makespan)
        
        # Constraint violation penalties
        precedence_violation = self.compute_precedence_violation(outputs, data)
        overlap_violation = self.compute_overlap_violation(outputs, data)
        duration_inconsistency = self.compute_duration_consistency(outputs)
        
        # Total loss
        total_loss = (
            self.processor_weight * processor_loss +
            self.start_weight * start_loss +
            self.end_weight * end_loss +
            self.makespan_weight * makespan_loss +
            self.precedence_penalty * precedence_violation +
            self.overlap_penalty * overlap_violation +
            self.duration_penalty * duration_inconsistency
        )
        
        return {
            'total': total_loss,
            'processor': processor_loss,
            'start': start_loss,
            'end': end_loss,
            'makespan': makespan_loss,
            'precedence_penalty': precedence_violation,
            'overlap_penalty': overlap_violation,
            'duration_penalty': duration_inconsistency
        }


def create_constraint_aware_model(**kwargs):
    """Create constraint-aware GNN model"""
    return ConstraintAwareGNN(**kwargs)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Constraint-Aware GNN Model')
    parser.add_argument('--data_path', type=str, default='training_data_multitask.pt', 
                        help='Path to the training data file')
    parser.add_argument('--output_dir', type=str, default='.', 
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=25, 
                        help='Number of training epochs')
    args = parser.parse_args()

    print("="*70)
    print("CONSTRAINT-AWARE GNN MODEL")
    print("="*70)
    print("\nModel jo constraints ke saath train hota hai:")
    print("✓ Eligibility masking in forward pass")
    print("✓ Precedence-aware start time prediction")
    print("✓ Non-overlap penalty in loss")
    print("✓ Duration consistency enforcement")
    print("="*70)
    print("CONSTRAINT-AWARE GNN MODEL")
    print("="*70)
    print("\nModel jo constraints ke saath train hota hai:")
    print("✓ Eligibility masking in forward pass")
    print("✓ Precedence-aware start time prediction")
    print("✓ Non-overlap penalty in loss")
    print("✓ Duration consistency enforcement")

    # TRAINING LOOP (DEMO)
    import torch.optim as optim
    from torch_geometric.loader import DataLoader
    # Load single multitask tensor file
    dataset = torch.load(args.data_path, weights_only=False)
    if not isinstance(dataset, list):
        dataset = [dataset]
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = create_constraint_aware_model()
    loss_fn = ConstraintAwareLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = args.epochs
    print(f"Training for {epochs} epochs on {len(dataset)} samples...")
    device = torch.device('cuda')
    model = model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        metrics = {'processor':0, 'start':0, 'end':0, 'makespan':0, 'precedence_penalty':0, 'overlap_penalty':0, 'duration_penalty':0}
        for batch in loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            can_run_on_mask = getattr(batch, 'can_run_on_mask', None)
            outputs = model(batch, can_run_on_masks=can_run_on_mask, enforce_constraints=True)
            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for k in metrics:
                val = loss_dict[k] if k in loss_dict else 0
                metrics[k] += val.item() if hasattr(val, 'item') else val
        n = len(loader)
        print(f"Epoch {epoch:3d} | Total Loss: {total_loss/n:.4f} | "
              f"Proc: {metrics['processor']/n:.3f} | Start: {metrics['start']/n:.3f} | End: {metrics['end']/n:.3f} | "
              f"Makespan: {metrics['makespan']/n:.3f} | Prec: {metrics['precedence_penalty']/n:.3f} | "
              f"Overlap: {metrics['overlap_penalty']/n:.3f} | Dur: {metrics['duration_penalty']/n:.3f}")

    # Save the trained model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'gnn_model_constrained.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
