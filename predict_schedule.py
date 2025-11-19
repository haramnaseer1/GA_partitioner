"""
GNN Schedule Prediction Script
==============================

Simple CLI for clients:
- Input: Application JSON file
- Output: Predicted schedule (node_id, start_time, end_time, dependencies)
- Uses constraint-aware GNN for valid predictions
"""

import torch
import json
import sys
from pathlib import Path
from train_gnn_constrained import create_constraint_aware_model
from torch_geometric.data import Data


def load_application(app_path):
    """Load application JSON"""
    with open(app_path, 'r') as f:
        return json.load(f)['application']


def create_graph(app_data):
    """Create PyG Data object from application"""
    jobs = app_data['jobs']
    messages = app_data['messages']
    num_tasks = len(jobs)
    
    # Node features: [processing_time, deadline, num_dependencies]
    dep_count = {job['id']: 0 for job in jobs}
    for msg in messages:
        dep_count[msg['receiver']] += 1
    node_features = []
    for job in jobs:
        node_features.append([
            float(job['processing_times']),
            float(job.get('deadline', 10000)),
            float(dep_count[job['id']])
        ])
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge index and attributes
    edge_index = []
    edge_attr = []
    for msg in messages:
        edge_index.append([msg['sender'], msg['receiver']])
        edge_attr.append([float(msg.get('size', 24))])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(num_tasks, dtype=torch.long)
    )
    return data


def create_eligibility_mask(app_data, num_processors=192):
    """Create eligibility mask for each task"""
    jobs = app_data['jobs']
    # Resource mapping (same as GA)
    resource_mapping = {
        1: 'General purpose CPU',
        2: 'FPGA',
        3: 'Raspberry Pi 5',
        4: 'Microcontroller',
        5: 'High Performance CPU',
        6: 'Graphical Processing Unit'
    }
    # Dummy processor types (for demo)
    processor_types = [resource_mapping[1]] * num_processors
    mask = torch.zeros((len(jobs), num_processors))
    for i, job in enumerate(jobs):
        can_run_on = job.get('can_run_on', [1])
        allowed_types = [resource_mapping[rt] for rt in can_run_on]
        for proc_id in range(num_processors):
            if processor_types[proc_id] in allowed_types:
                mask[i, proc_id] = 1.0
    return mask


def predict_schedule(app_path, model_path='models/gnn_model_constrained.pth'):
    """Predict schedule for input application"""
    print(f"\n=== GNN Schedule Prediction ===")
    print(f"Input: {app_path}")
    
    # Load application
    app_data = load_application(app_path)
    num_processors = 192  # Hardcoded in create_eligibility_mask
    data = create_graph(app_data)
    eligibility_mask = create_eligibility_mask(app_data)
    
    # Load model
    model = create_constraint_aware_model(num_processors=num_processors)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove incompatible processor_head layers
    state_dict.pop('processor_head.4.weight', None)
    state_dict.pop('processor_head.4.bias', None)
    
    model.load_state_dict(state_dict, strict=False)
    print(f"✓ Loaded trained model (num_processors={num_processors}, skipped incompatible layers)\n")
    model.eval()
    
    # Predict
    with torch.no_grad():
        outputs = model(data, can_run_on_masks=eligibility_mask, enforce_constraints=True)
    
    # Build output schedule
    jobs = app_data['jobs']
    schedule = []
    processor_assignments = outputs['processor'].argmax(dim=1)
    for i, job in enumerate(jobs):
        schedule.append({
            'task_id': job['id'],
            'node_id': int(processor_assignments[i].item()),
            'start_time': float(outputs['start_time'][i].item()),
            'end_time': float(outputs['end_time'][i].item()),
            'dependencies': [
                {'task_id': msg['sender'], 'message_size': msg.get('size', 24)}
                for msg in app_data['messages'] if msg['receiver'] == job['id']
            ]
        })
    
    # Sort for readability
    schedule.sort(key=lambda x: (x['node_id'], x['start_time'], x['task_id']))
    
    # Print output
    print(f"\nPredicted Schedule:")
    for row in schedule:
        print(f"Task {row['task_id']:2d} | Node {row['node_id']:3d} | Start {row['start_time']:.2f} | End {row['end_time']:.2f} | Deps: {row['dependencies']}")

    # Calculate and print makespan
    makespan = max(row['end_time'] for row in schedule)
    print(f"\nPredicted Makespan: {makespan:.2f}")
    
    # Optionally save to file
    out_path = Path(app_path).stem + '_gnn_schedule.json'
    with open(out_path, 'w') as f:
        json.dump(schedule, f, indent=2)
    print(f"\n✓ Saved to {out_path}")
    return schedule


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_schedule.py <Application/T2.json>")
        sys.exit(1)
    app_path = sys.argv[1]
    predict_schedule(app_path)
