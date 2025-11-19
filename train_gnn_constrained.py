import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import argparse
import os
import sys


class ConstraintAwareGNN(nn.Module):
    """
    Platform-aware GNN that learns scheduling with proper constraint signals
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
        self.app_constraints = app_constraints
        
        # Platform encoding
        if platform_info is not None:
            self.register_platform_info(platform_info)
        else:
            self.register_buffer('processor_speeds', torch.ones(num_processors))
            self.register_buffer('processor_tiers', torch.zeros(num_processors, dtype=torch.long))
            self.register_buffer('processor_locations', torch.zeros(num_processors, 3))
        
        # Platform encoder: [speed, tier_onehot(3), location(3)] -> hidden
        platform_feature_dim = 1 + 3 + 3
        self.platform_encoder = nn.Sequential(
            nn.Linear(platform_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
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
        
        # Task projection for bilinear scoring
        self.task_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Processor projection
        self.proc_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # START TIME PREDICTION (raw, uncorrected)
        self.start_time_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # COMMUNICATION DELAY (soft, differentiable)
        self.comm_delay_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable tier delay matrix [3, 3] for all tier pairs
        self.tier_delay_matrix = nn.Parameter(torch.tensor([
            [50.0, 150.0, 350.0],
            [150.0, 50.0, 200.0],
            [350.0, 200.0, 175.0]
        ]))
    
    def register_platform_info(self, platform_info):
        """Register platform topology"""
        self.register_buffer('processor_speeds', 
                           torch.tensor(platform_info['processor_speeds'], dtype=torch.float32))
        self.register_buffer('processor_tiers', 
                           torch.tensor(platform_info['processor_tiers'], dtype=torch.long))
        self.register_buffer('processor_locations', 
                           torch.tensor(platform_info['processor_locations'], dtype=torch.float32))
    
    def get_platform_features(self):
        """Create per-processor feature vectors"""
        speeds = self.processor_speeds.unsqueeze(-1)
        tiers_onehot = F.one_hot(self.processor_tiers, num_classes=3).float()
        locations = self.processor_locations
        platform_features = torch.cat([speeds, tiers_onehot, locations], dim=-1)
        return platform_features
    
    def compute_duration_physics_based(self, task_processing_times, processor_probs):
        """Physics-based: duration = processing_time / clock_speed"""
        expected_speeds = torch.matmul(processor_probs, self.processor_speeds)
        expected_speeds = expected_speeds.unsqueeze(-1)
        durations = task_processing_times / (expected_speeds + 1e-6)
        return durations
    
    def get_soft_tier_delay(self, sender_tier_probs, receiver_tier_probs):
        """Compute expected communication delay using SOFT tier probabilities"""
        delays = []
        for i in range(sender_tier_probs.size(0)):
            s_probs = sender_tier_probs[i]
            r_probs = receiver_tier_probs[i]
            delay = torch.sum(s_probs.unsqueeze(1) * self.tier_delay_matrix * r_probs.unsqueeze(0))
            delays.append(delay)
        return torch.stack(delays).unsqueeze(-1)
    
    def apply_eligibility_mask(self, processor_logits, can_run_on_mask):
        """Apply eligibility constraint"""
        masked_logits = processor_logits.clone()
        masked_logits = masked_logits + (-1e9) * (1 - can_run_on_mask)
        return masked_logits
    
    def compute_precedence_constraints(self, node_features, edge_index, edge_attr, 
                                      processor_probs, base_start_times, durations):
        """
        Compute precedence-aware start times using SOFT processor assignments
        FULLY DIFFERENTIABLE - NO IN-PLACE OPERATIONS
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        if edge_index.size(1) == 0:
            return base_start_times
        
        sender_indices = edge_index[0]
        receiver_indices = edge_index[1]
        
        sender_features = node_features[sender_indices]
        receiver_features = node_features[receiver_indices]
        
        # Prepare edge attributes
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        elif edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 1, device=device)
        
        # Get SOFT tier probabilities
        tiers_onehot = F.one_hot(self.processor_tiers, num_classes=3).float()
        tier_probs = processor_probs @ tiers_onehot
        
        sender_tier_probs = tier_probs[sender_indices]
        receiver_tier_probs = tier_probs[receiver_indices]
        
        # Soft tier-based delay
        tier_delay_base = self.get_soft_tier_delay(sender_tier_probs, receiver_tier_probs)
        
        # Learned delay adjustment
        combined = torch.cat([
            sender_features, 
            receiver_features, 
            edge_attr,
            sender_tier_probs,
            receiver_tier_probs
        ], dim=-1)
        
        delay_adjustment = self.comm_delay_head(combined)
        comm_delays = F.relu(tier_delay_base + delay_adjustment)
        
        # Calculate precedence constraints
        sender_starts = base_start_times[sender_indices]
        sender_durations = durations[sender_indices]
        sender_ends = sender_starts + sender_durations
        
        min_receiver_starts = sender_ends + comm_delays
        
        # DIFFERENTIABLE SCATTER-MAX (NO IN-PLACE OPERATIONS)
        adjusted_start_times = base_start_times.clone()
        
        # Create mask for each node: [num_nodes, num_edges]
        receiver_mask = (receiver_indices.unsqueeze(0) == torch.arange(num_nodes, device=device).unsqueeze(1)).float()
        
        # Apply mask: constraints only for valid receiver-edge pairs
        constraints_per_node = receiver_mask * min_receiver_starts.squeeze(-1).unsqueeze(0)
        
        # For nodes with no constraints, set to -inf
        constraints_per_node = torch.where(
            receiver_mask > 0,
            constraints_per_node,
            torch.tensor(float('-inf'), device=device)
        )
        
        # Element-wise maximum along edge dimension
        max_constraints = torch.amax(constraints_per_node, dim=1, keepdim=True)
        
        # Replace -inf with base start times
        max_constraints = torch.where(
            torch.isinf(max_constraints),
            base_start_times,
            max_constraints
        )
        
        adjusted_start_times = torch.max(base_start_times, max_constraints)
        
        return adjusted_start_times
    
    def forward(self, data, can_run_on_masks=None, enforce_constraints=True):
        """Forward pass with proper constraint handling"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        device = x.device
        
        # Extract task processing time
        task_processing_times = x[:, 0:1]
        
        # Encode task features
        x_encoded = self.node_encoder(x)
        edge_attr = self.edge_encoder(data.edge_attr) if data.edge_attr is not None else None
        
        # Encode platform (per-processor)
        platform_features = self.get_platform_features()
        platform_encoded = self.platform_encoder(platform_features)
        
        # GAT layers
        x = x_encoded
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_new = gat(x, edge_index, edge_attr=edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=0.2, training=self.training)
            
            if x.size(-1) == x_new.size(-1) and i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # PROCESSOR ASSIGNMENT - TRUE PER-PROCESSOR AWARENESS
        task_emb = self.task_proj(x)
        proc_emb = self.proc_proj(platform_encoded)
        
        processor_logits = torch.matmul(task_emb, proc_emb.T)
        
        if enforce_constraints and can_run_on_masks is not None:
            can_run_on_masks = can_run_on_masks.to(device)
            processor_logits = self.apply_eligibility_mask(processor_logits, can_run_on_masks)
        
        processor_probs = F.softmax(processor_logits, dim=-1)
        
        # DURATION - PHYSICS-BASED
        durations = self.compute_duration_physics_based(task_processing_times, processor_probs)
        durations = F.relu(durations) + 1e-3
        
        # START TIME - RAW and CORRECTED
        base_start_times = F.relu(self.start_time_base(x))
        
        if enforce_constraints and edge_index.size(1) > 0:
            start_times_corrected = self.compute_precedence_constraints(
                x, edge_index, edge_attr, processor_probs, base_start_times, durations
            )
        else:
            start_times_corrected = base_start_times
        
        # END TIMES
        end_times_raw = base_start_times + durations
        end_times_corrected = start_times_corrected + durations
        
        # MAKESPAN
        if batch is not None:
            unique_batches = torch.unique(batch)
            makespans = []
            for b in unique_batches:
                batch_mask = (batch == b)
                batch_max = end_times_corrected[batch_mask].max()
                makespans.append(batch_max)
            makespan = torch.stack(makespans)
        else:
            makespan = end_times_corrected.max().unsqueeze(0)
        
        return {
            'processor': processor_logits,
            'processor_probs': processor_probs,
            'start_time_raw': base_start_times,
            'start_time': start_times_corrected,
            'end_time_raw': end_times_raw,
            'end_time': end_times_corrected,
            'duration': durations,
            'makespan': makespan
        }


class ConstraintAwareLoss(nn.Module):
    """Loss function with PROPER constraint penalties"""
    
    def __init__(self, 
                 processor_weight=1.0,
                 start_weight=1.0,
                 end_weight=1.0,
                 makespan_weight=1.0,
                 precedence_penalty=10.0,
                 overlap_penalty=10.0,
                 duration_penalty=0.0):
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
        """Precedence penalty using RAW predictions (not corrected)"""
        if data.edge_index.size(1) == 0:
            return torch.tensor(0.0, device=outputs['start_time_raw'].device)
        
        sender_indices = data.edge_index[0]
        receiver_indices = data.edge_index[1]
        
        sender_ends = outputs['end_time_raw'][sender_indices]
        receiver_starts = outputs['start_time_raw'][receiver_indices]
        
        violations = F.relu(sender_ends - receiver_starts)
        
        return violations.mean()
    
    def compute_overlap_violation(self, outputs, data):
        """Non-overlap penalty - BATCH AWARE"""
        processor_assignments = outputs['processor'].argmax(dim=1)
        device = processor_assignments.device
        num_tasks = outputs['start_time_raw'].size(0)
        
        if num_tasks <= 1:
            return torch.tensor(0.0, device=device)
        
        batch = data.batch
        
        proc_i = processor_assignments.unsqueeze(1)
        proc_j = processor_assignments.unsqueeze(0)
        same_proc = (proc_i == proc_j).float()
        
        batch_i = batch.unsqueeze(1)
        batch_j = batch.unsqueeze(0)
        same_graph = (batch_i == batch_j).float()
        
        mask = torch.triu(torch.ones(num_tasks, num_tasks, device=device), diagonal=1)
        
        same_proc_same_graph = same_proc * same_graph * mask
        
        if same_proc_same_graph.sum() < 1e-6:
            return torch.tensor(0.0, device=device)
        
        start_i = outputs['start_time_raw'].squeeze(-1).unsqueeze(1)
        end_i = outputs['end_time_raw'].squeeze(-1).unsqueeze(1)
        start_j = outputs['start_time_raw'].squeeze(-1).unsqueeze(0)
        end_j = outputs['end_time_raw'].squeeze(-1).unsqueeze(0)
        
        overlap = F.relu(torch.min(end_i, end_j) - torch.max(start_i, start_j))
        violations = (overlap * same_proc_same_graph).sum() / (same_proc_same_graph.sum() + 1e-8)
        
        return violations
    
    def compute_duration_consistency(self, outputs, data):
        """Duration consistency with ground truth (if available)"""
        if hasattr(data, 'y_duration') and self.duration_penalty > 0:
            return self.regression_loss_fn(outputs['duration'].squeeze(-1), data.y_duration)
        else:
            return torch.tensor(0.0, device=outputs['duration'].device)
    
    def forward(self, outputs, data):
        """Compute total loss"""
        device = outputs['processor'].device
        
        processor_loss = self.processor_loss_fn(outputs['processor'], data.y_processor)
        start_loss = self.regression_loss_fn(outputs['start_time'].squeeze(-1), data.y_start)
        end_loss = self.regression_loss_fn(outputs['end_time'].squeeze(-1), data.y_end)
        
        makespan_pred = outputs['makespan']
        
        if hasattr(data, 'y_makespan'):
            makespan_target = data.y_makespan
            
            if makespan_target.dim() == 0:
                makespan_target = makespan_target.unsqueeze(0)
            elif makespan_target.dim() == 2:
                makespan_target = makespan_target.squeeze(-1)
            
            if makespan_target.size(0) == 1 and makespan_pred.size(0) > 1:
                makespan_target = makespan_target.expand(makespan_pred.size(0))
            elif makespan_target.size(0) != makespan_pred.size(0):
                min_size = min(makespan_target.size(0), makespan_pred.size(0))
                makespan_target = makespan_target[:min_size]
                makespan_pred = makespan_pred[:min_size]
            
            makespan_loss = self.regression_loss_fn(makespan_pred, makespan_target)
        else:
            makespan_loss = torch.tensor(0.0, device=device)
        
        precedence_violation = self.compute_precedence_violation(outputs, data)
        overlap_violation = self.compute_overlap_violation(outputs, data)
        
        if self.duration_penalty > 0:
            duration_inconsistency = self.compute_duration_consistency(outputs, data)
        else:
            duration_inconsistency = torch.tensor(0.0, device=device)
        
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
    """Create platform-aware constraint GNN model"""
    return ConstraintAwareGNN(**kwargs)


def create_example_platform_info(num_processors=192):
    """Create example 3-tier platform"""
    platform_info = {
        'processor_speeds': [],
        'processor_tiers': [],
        'processor_locations': []
    }
    
    for i in range(num_processors):
        if i < 64:  # Edge
            platform_info['processor_speeds'].append(2.0)
            platform_info['processor_tiers'].append(0)
            platform_info['processor_locations'].append([i % 8, i // 8, 0])
        elif i < 128:  # Fog
            platform_info['processor_speeds'].append(3.0)
            platform_info['processor_tiers'].append(1)
            platform_info['processor_locations'].append([(i-64) % 4 + 2, (i-64) // 4 + 2, 1])
        else:  # Cloud
            platform_info['processor_speeds'].append(4.0)
            platform_info['processor_tiers'].append(2)
            platform_info['processor_locations'].append([4, 4, 2])
    
    return platform_info


def train_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch with progress indicators"""
    model.train()
    total_loss = 0.0
    metrics = {
        'processor': 0, 'start': 0, 'end': 0, 'makespan': 0,
        'precedence_penalty': 0, 'overlap_penalty': 0
    }
    
    num_batches = len(loader)
    
    for batch_idx, batch in enumerate(loader):
        print(f"  [Train] Batch {batch_idx + 1}/{num_batches}...", end='\r', flush=True)
        sys.stdout.flush()
        
        optimizer.zero_grad()
        batch = batch.to(device)
        
        can_run_on_mask = getattr(batch, 'can_run_on_mask', None)
        outputs = model(batch, can_run_on_masks=can_run_on_mask, enforce_constraints=True)
        
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict['total']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k in metrics:
            if k in loss_dict:
                metrics[k] += loss_dict[k].item()
    
    n = len(loader)
    avg_metrics = {k: v / n for k, v in metrics.items()}
    avg_metrics['total'] = total_loss / n
    
    print(" " * 60, end='\r', flush=True)
    
    return avg_metrics


def validate_epoch(model, loader, loss_fn, device):
    """Validate for one epoch with progress indicators"""
    model.eval()
    total_loss = 0.0
    metrics = {
        'processor': 0, 'start': 0, 'end': 0, 'makespan': 0,
        'precedence_penalty': 0, 'overlap_penalty': 0
    }
    
    num_batches = len(loader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(f"  [Val]   Batch {batch_idx + 1}/{num_batches}...", end='\r', flush=True)
            sys.stdout.flush()
            
            batch = batch.to(device)
            
            can_run_on_mask = getattr(batch, 'can_run_on_mask', None)
            outputs = model(batch, can_run_on_masks=can_run_on_mask, enforce_constraints=True)
            
            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict['total']
            
            total_loss += loss.item()
            for k in metrics:
                if k in loss_dict:
                    metrics[k] += loss_dict[k].item()
    
    n = len(loader)
    avg_metrics = {k: v / n for k, v in metrics.items()}
    avg_metrics['total'] = total_loss / n
    
    print(" " * 60, end='\r', flush=True)
    
    return avg_metrics


def save_checkpoint(model, optimizer, epoch, metrics, platform_info, output_dir, 
                   filename='checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'platform_info': platform_info,
        'model_config': {
            'hidden_dim': model.hidden_dim,
            'num_processors': model.num_processors
        }
    }
    
    filepath = os.path.join(output_dir, filename)
    torch.save(checkpoint, filepath)
    return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Platform-Aware Constraint GNN')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--platform_file', type=str, default=None)
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--best_metric', type=str, default='makespan',
                       choices=['makespan', 'precedence_penalty', 'total', 'processor'],
                       help='Metric for best model selection')
    args = parser.parse_args()

    print("="*70)
    print("PLATFORM-AWARE CONSTRAINT GNN (FULLY FIXED)")
    print("="*70)
    print("\n✓ FIXED ISSUES:")
    print("  • Duration penalty uses ground truth (or disabled)")
    print("  • Precedence penalty uses RAW start times")
    print("  • Overlap penalty is batch-aware")
    print("  • Soft processor assignments (no argmax in constraints)")
    print("  • True per-processor embeddings for assignment")
    print("  • DIFFERENTIABLE SCATTER-MAX (no in-place ops)")
    print("  • PROPER OUTPUT FLUSHING with progress indicators")
    print("\n✓ PLATFORM FEATURES:")
    print("  • Per-processor speed/tier/location encoding")
    print("  • Physics-based duration: time / speed")
    print("  • Differentiable tier-aware communication delays")
    print("  • Bilinear task-processor scoring")
    print("\n✓ TRAINING FEATURES:")
    print("  • Train/validation split for better generalization")
    print("  • Best model tracking (metric: {})".format(args.best_metric))
    print("  • Checkpoint saving for resume training")
    print("="*70)

    import torch.optim as optim
    from torch_geometric.loader import DataLoader
    import random
    import json
    
    # Load dataset
    dataset = torch.load(args.data, weights_only=False)
    if not isinstance(dataset, list):
        dataset = [dataset]
    
    # Train/Val split
    random.seed(42)
    random.shuffle(dataset)
    
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Platform info
    if args.platform_file:
        with open(args.platform_file, 'r') as f:
            platform_info = json.load(f)
    else:
        platform_info = create_example_platform_info(192)
    
    print(f"\nPlatform: {len(platform_info['processor_speeds'])} processors")
    print(f"  Edge: {sum(1 for t in platform_info['processor_tiers'] if t == 0)}")
    print(f"  Fog: {sum(1 for t in platform_info['processor_tiers'] if t == 1)}")
    print(f"  Cloud: {sum(1 for t in platform_info['processor_tiers'] if t == 2)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_constraint_aware_model(platform_info=platform_info).to(device)
    loss_fn = ConstraintAwareLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Resume training
    start_epoch = 1
    best_val_metric = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_metric' in checkpoint['metrics']:
            best_val_metric = checkpoint['metrics']['best_val_metric']
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best validation {args.best_metric}: {best_val_metric:.4f}")
    
    print(f"\nTraining: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"Device: {device}")
    print(f"Best metric: {args.best_metric}")
    print(f"Output directory: {args.output}\n")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch:3d}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        print("Training...")
        sys.stdout.flush()
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Validate
        print("Validating...")
        sys.stdout.flush()
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)
        
        # Print metrics
        print(f"\nEpoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['total']:.4f} | "
              f"Val Loss: {val_metrics['total']:.4f}")
        print(f"    Train - Proc: {train_metrics['processor']:.3f} | "
              f"Start: {train_metrics['start']:.3f} | "
              f"End: {train_metrics['end']:.3f} | "
              f"Makespan: {train_metrics['makespan']:.3f}")
        print(f"    Train - Prec: {train_metrics['precedence_penalty']:.3f} | "
              f"Overlap: {train_metrics['overlap_penalty']:.3f}")
        print(f"    Val   - Proc: {val_metrics['processor']:.3f} | "
              f"Start: {val_metrics['start']:.3f} | "
              f"End: {val_metrics['end']:.3f} | "
              f"Makespan: {val_metrics['makespan']:.3f}")
        print(f"    Val   - Prec: {val_metrics['precedence_penalty']:.3f} | "
              f"Overlap: {val_metrics['overlap_penalty']:.3f}")
        
        sys.stdout.flush()
        
        # Check if this is the best model
        current_val_metric = val_metrics[args.best_metric]
        is_best = current_val_metric < best_val_metric
        
        if is_best:
            best_val_metric = current_val_metric
            print(f"    ★ New best {args.best_metric}: {best_val_metric:.4f}")
            
            # Save best model
            best_path = save_checkpoint(
                model, optimizer, epoch, 
                {'train': train_metrics, 'val': val_metrics, 'best_val_metric': best_val_metric},
                platform_info, args.output, 
                filename='best_model.pth'
            )
            print(f"    ★ Best model saved: {best_path}")
        
        # Save last checkpoint (for resuming)
        if epoch % 5 == 0 or epoch == args.epochs:
            last_path = save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics, 'best_val_metric': best_val_metric},
                platform_info, args.output,
                filename='last_checkpoint.pth'
            )
            print(f"    💾 Checkpoint saved: {last_path}")
        
        sys.stdout.flush()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best validation {args.best_metric}: {best_val_metric:.4f}")
    print(f"\nSaved models:")
    print(f"  • Best model: {os.path.join(args.output, 'best_model.pth')}")
    print(f"  • Last checkpoint: {os.path.join(args.output, 'last_checkpoint.pth')}")
    print("\nTo resume training:")
    print(f"  python {os.path.basename(__file__)} --data {args.data} --resume {os.path.join(args.output, 'last_checkpoint.pth')} --epochs {args.epochs + 50}")
    print("="*70)