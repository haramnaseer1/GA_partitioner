"""
GNN Tensor Conversion Script
=============================

Converts preprocessed scheduling data into PyTorch Geometric graph tensors.

This script creates graph representations suitable for GNN training:
- Node features: task characteristics (processing time, deadline, etc.)
- Edge features: message dependencies (size, communication overhead)
- Labels: optimal node assignments from GA solutions

Input:
    - gnn_solution/preprocessed_data.pkl

Output:
    - gnn_solution/graph_dataset.pt (PyTorch Geometric dataset)
    - gnn_solution/dataset_info.json (metadata)

Usage:
    python Script/convert_to_gnn_tensors.py
"""

import os
import json
import pickle
import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.utils import from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️  Warning: torch-geometric not installed. Install with: pip install torch-geometric")


class SchedulingGraphDataset:
    """
    Custom dataset for scheduling graphs.
    Each sample is a Data object with:
        - x: node features [num_tasks, feature_dim]
        - edge_index: edge connectivity [2, num_edges]
        - edge_attr: edge features [num_edges, edge_feature_dim]
        - y: node assignment labels [num_tasks]
        - platform_features: platform node features [num_platform_nodes, platform_feature_dim]
    """
    
    def __init__(self, preprocessed_data: Dict):
        """
        Args:
            preprocessed_data: Output from preprocess_gnn_data.py
        """
        self.samples = preprocessed_data['samples']
        self.feature_stats = preprocessed_data['feature_stats']
        self.metadata = preprocessed_data['metadata']
        self.graphs = []
        
        self._convert_all_samples()
    
    def _convert_all_samples(self):
        """Convert all samples to PyTorch Geometric Data objects."""
        print(f"\nConverting {len(self.samples)} samples to graph tensors...")
        
        for idx, sample in enumerate(self.samples, 1):
            try:
                graph_data = self._sample_to_graph(sample)
                self.graphs.append(graph_data)
                
                if idx % 20 == 0:
                    print(f"  [{idx}/{len(self.samples)}] Converted {sample['app_name']}")
            except Exception as e:
                print(f"  ERROR converting {sample['app_name']}: {str(e)}")
                continue
        
        print(f"✅ Converted {len(self.graphs)} graphs successfully")
    
    def _sample_to_graph(self, sample: Dict) -> Data:
        """
        Convert a single sample to PyTorch Geometric Data object.
        
        Args:
            sample: Preprocessed application sample
        
        Returns:
            Data object with node/edge features and labels
        """
        num_jobs = sample['num_jobs']
        jobs = sample['jobs']
        messages = sample['messages']
        solution = sample['solution']
        
        # ==================== Node Features ====================
        # Features per task node:
        # [processing_time_norm, deadline_norm, num_dependencies, out_degree]
        # + one-hot encoding of can_run_on (6 resource types)
        
        node_features = []
        task_degrees = self._compute_task_degrees(jobs, messages)
        
        for task_id in range(num_jobs):
            job = jobs[task_id]
            
            # Basic features
            features = [
                job['processing_time_norm'],  # Normalized processing time
                job.get('deadline', 10000) / 10000.0,  # Normalized deadline
                task_degrees[task_id]['in_degree'],  # Number of dependencies
                task_degrees[task_id]['out_degree'],  # Number of successors
            ]
            
            # One-hot encode resource compatibility (can_run_on)
            resource_one_hot = [0] * 6  # 6 resource types
            for resource_id in job.get('can_run_on', []):
                if 1 <= resource_id <= 6:
                    resource_one_hot[resource_id - 1] = 1
            
            features.extend(resource_one_hot)
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # ==================== Edge Index & Features ====================
        # Edge index: [2, num_edges] format (source, target)
        # Edge features: [message_size_norm, communication_overhead]
        
        edge_list = []
        edge_features = []
        
        for msg_id, msg in messages.items():
            sender = msg['sender']
            receiver = msg['receiver']
            
            # Add edge (directed: sender -> receiver)
            edge_list.append([sender, receiver])
            
            # Edge features
            edge_feat = [
                msg['size_norm'],  # Normalized message size
                msg['size'] / 1000.0,  # Message size in KB (normalized)
            ]
            edge_features.append(edge_feat)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # No edges (independent tasks)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        # ==================== Labels ====================
        # y: node assignment for each task (which platform node executes it)
        
        task_to_node = {}
        for task_assignment in solution:
            task_id = task_assignment['task_id']
            node_id = task_assignment['node_id']
            task_to_node[task_id] = node_id
        
        # Create label vector
        labels = []
        for task_id in range(num_jobs):
            node_id = task_to_node.get(task_id, 0)  # Default to 0 if missing
            labels.append(node_id)
        
        y = torch.tensor(labels, dtype=torch.long)
        
        # ==================== Platform Features ====================
        # Additional context: platform node features
        # [clock_speed_norm, resource_type_id, tier, is_router]
        
        platform_nodes = sample['platform_nodes']
        platform_features = []
        platform_node_ids = []
        
        for node_id, node_info in platform_nodes.items():
            platform_node_ids.append(node_id)
            platform_features.append([
                node_info['clock_speed_norm'],
                node_info['resource_type_id'],
                node_info['tier'],  # 0=Edge, 1=Fog, 2=Cloud
                1.0 if node_info['is_router'] else 0.0
            ])
        
        platform_x = torch.tensor(platform_features, dtype=torch.float)
        platform_ids = torch.tensor(platform_node_ids, dtype=torch.long)
        
        # ==================== Create Data Object ====================
        data = Data(
            x=node_features,              # Task node features [num_tasks, 10]
            edge_index=edge_index,        # Edge connectivity [2, num_edges]
            edge_attr=edge_attr,          # Edge features [num_edges, 2]
            y=y,                          # Node assignments [num_tasks]
            platform_x=platform_x,        # Platform features [num_platform_nodes, 4]
            platform_ids=platform_ids,    # Platform node IDs
            num_tasks=num_jobs,           # Metadata
            num_messages=len(messages),
            app_name=sample['app_name']
        )
        
        return data
    
    def _compute_task_degrees(self, jobs: Dict, messages: Dict) -> Dict:
        """
        Compute in-degree and out-degree for each task.
        
        Returns:
            Dict[task_id] = {'in_degree': int, 'out_degree': int}
        """
        degrees = {task_id: {'in_degree': 0, 'out_degree': 0} 
                   for task_id in jobs.keys()}
        
        for msg in messages.values():
            sender = msg['sender']
            receiver = msg['receiver']
            
            if sender in degrees:
                degrees[sender]['out_degree'] += 1
            if receiver in degrees:
                degrees[receiver]['in_degree'] += 1
        
        return degrees
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def get_feature_dimensions(self) -> Dict:
        """Get dimensions of all feature vectors."""
        if not self.graphs:
            return {}
        
        sample_graph = self.graphs[0]
        return {
            'node_feature_dim': sample_graph.x.shape[1] if sample_graph.x.numel() > 0 else 0,
            'edge_feature_dim': sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.numel() > 0 else 0,
            'platform_feature_dim': sample_graph.platform_x.shape[1] if sample_graph.platform_x.numel() > 0 else 0,
            'num_classes': len(sample_graph.platform_ids)  # Number of platform nodes
        }


def save_dataset(dataset: SchedulingGraphDataset, output_dir: str):
    """Save dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graphs as PyTorch file
    output_path = os.path.join(output_dir, 'graph_dataset.pt')
    torch.save({
        'graphs': dataset.graphs,
        'feature_stats': dataset.feature_stats,
        'metadata': dataset.metadata
    }, output_path)
    
    print(f"\n✅ Saved graph dataset: {output_path}")
    
    # Save dataset info
    feature_dims = dataset.get_feature_dimensions()
    dataset_info = {
        'num_samples': len(dataset),
        'feature_dimensions': feature_dims,
        'metadata': dataset.metadata,
        'sample_statistics': {
            'avg_num_tasks': np.mean([g.num_tasks for g in dataset.graphs]),
            'avg_num_edges': np.mean([g.edge_index.shape[1] for g in dataset.graphs]),
            'min_tasks': min([g.num_tasks for g in dataset.graphs]),
            'max_tasks': max([g.num_tasks for g in dataset.graphs]),
        }
    }
    
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Saved dataset info: {info_path}")
    
    return dataset_info


def create_train_test_split(dataset: SchedulingGraphDataset, 
                            train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    Split dataset into train and test sets.
    
    Args:
        dataset: SchedulingGraphDataset instance
        train_ratio: Ratio of training samples (default: 0.8)
    
    Returns:
        (train_graphs, test_graphs)
    """
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    split_idx = int(num_samples * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_graphs = [dataset.graphs[i] for i in train_indices]
    test_graphs = [dataset.graphs[i] for i in test_indices]
    
    print(f"\n📊 Dataset Split:")
    print(f"  Training samples: {len(train_graphs)} ({train_ratio*100:.0f}%)")
    print(f"  Testing samples: {len(test_graphs)} ({(1-train_ratio)*100:.0f}%)")
    
    return train_graphs, test_graphs


def main():
    """Main conversion pipeline."""
    print("="*80)
    print("GNN Tensor Conversion")
    print("="*80)
    
    # Check PyTorch Geometric installation
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("\n❌ PyTorch Geometric is required!")
        print("Install with: pip install torch-geometric")
        return
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    input_pkl = os.path.join(repo_root, 'gnn_solution', 'preprocessed_data.pkl')
    output_dir = os.path.join(repo_root, 'gnn_solution')
    
    # Check input file exists
    if not os.path.exists(input_pkl):
        print(f"\n❌ Preprocessed data not found: {input_pkl}")
        print("Run preprocessing first: python Script/preprocess_gnn_data.py")
        return
    
    # Load preprocessed data
    print(f"\nLoading preprocessed data from: {input_pkl}")
    with open(input_pkl, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    print(f"✅ Loaded {len(preprocessed_data['samples'])} samples")
    
    # Convert to graph tensors
    print("\nStep 1: Converting to graph tensors...")
    dataset = SchedulingGraphDataset(preprocessed_data)
    
    # Display sample graph info
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n📊 Sample Graph Structure ('{sample.app_name}'):")
        print(f"  Node features shape: {sample.x.shape}")
        print(f"  Edge index shape: {sample.edge_index.shape}")
        print(f"  Edge features shape: {sample.edge_attr.shape}")
        print(f"  Labels shape: {sample.y.shape}")
        print(f"  Platform features shape: {sample.platform_x.shape}")
    
    # Save dataset
    print("\nStep 2: Saving dataset...")
    dataset_info = save_dataset(dataset, output_dir)
    
    # Create train/test split
    print("\nStep 3: Creating train/test split...")
    train_graphs, test_graphs = create_train_test_split(dataset, train_ratio=0.8)
    
    # Save splits
    split_path = os.path.join(output_dir, 'train_test_split.pt')
    torch.save({
        'train': train_graphs,
        'test': test_graphs
    }, split_path)
    print(f"✅ Saved train/test split: {split_path}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ Conversion Complete!")
    print("="*80)
    print(f"\nDataset Summary:")
    print(f"  Total graphs: {len(dataset)}")
    print(f"  Node feature dim: {dataset_info['feature_dimensions']['node_feature_dim']}")
    print(f"  Edge feature dim: {dataset_info['feature_dimensions']['edge_feature_dim']}")
    print(f"  Platform feature dim: {dataset_info['feature_dimensions']['platform_feature_dim']}")
    print(f"  Average tasks per graph: {dataset_info['sample_statistics']['avg_num_tasks']:.1f}")
    print(f"  Average edges per graph: {dataset_info['sample_statistics']['avg_num_edges']:.1f}")
    print(f"  Task range: {dataset_info['sample_statistics']['min_tasks']} - {dataset_info['sample_statistics']['max_tasks']}")
    
    print(f"\nOutput files:")
    print(f"  - gnn_solution/graph_dataset.pt")
    print(f"  - gnn_solution/dataset_info.json")
    print(f"  - gnn_solution/train_test_split.pt")
    
    print(f"\n🚀 Ready for GNN training!")
    
    # Example usage code
    print(f"\n" + "="*80)
    print("Example: Load Dataset for Training")
    print("="*80)
    print("""
import torch
from torch_geometric.loader import DataLoader

# Load dataset
data = torch.load('gnn_solution/graph_dataset.pt')
graphs = data['graphs']

# Load train/test split
splits = torch.load('gnn_solution/train_test_split.pt')
train_graphs = splits['train']
test_graphs = splits['test']

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# Iterate over batches
for batch in train_loader:
    # batch.x: node features
    # batch.edge_index: graph connectivity
    # batch.y: labels (node assignments)
    predictions = model(batch)
    loss = criterion(predictions, batch.y)
    """)


if __name__ == "__main__":
    main()
