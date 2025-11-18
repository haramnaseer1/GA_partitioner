"""Convert GA solutions to PyTorch tensors for GNN training"""
import json
import torch
import numpy as np
from pathlib import Path
import glob

def load_solution(solution_path):
    """Load a solution JSON file"""
    with open(solution_path, 'r') as f:
        data = json.load(f)
        # Convert array format to schedule dict
        schedule = {}
        for task in data:
            task_id = task['task_id']
            processor = task['node_id']
            start = task['start_time']
            end = task['end_time']
            deps = task.get('dependencies', [])
            schedule[task_id] = (processor, start, end, deps)
        return {'schedule': schedule, 'makespan': max(task['end_time'] for task in data)}

def load_application(app_path):
    """Load application JSON file"""
    with open(app_path, 'r') as f:
        return json.load(f)['application']

def create_node_features(app_data, schedule):
    """Create node feature matrix for tasks
    Features: [processing_time, deadline, start_time, end_time, processor_id, is_scheduled]
    """
    jobs = {job['id']: job for job in app_data['jobs']}
    num_tasks = len(jobs)
    
    # Initialize features
    features = []
    for task_id in sorted(jobs.keys()):
        job = jobs[task_id]
        if task_id in schedule:
            processor, start, end, _ = schedule[task_id]
            scheduled = 1.0
        else:
            processor, start, end = 0, 0.0, 0.0
            scheduled = 0.0
        
        features.append([
            float(job['processing_times']),
            float(job['deadline']),
            float(start),
            float(end),
            float(processor),
            scheduled
        ])
    
    return torch.tensor(features, dtype=torch.float32)

def create_edge_index(app_data):
    """Create edge index (COO format) from message dependencies
    Returns: edge_index [2, num_edges], edge_attr [num_edges, edge_features]
    """
    messages = app_data['messages']
    
    edge_list = []
    edge_features = []
    
    for msg in messages:
        sender = msg['sender']
        receiver = msg['receiver']
        size = msg['size']
        
        edge_list.append([sender, receiver])
        edge_features.append([float(size)])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    
    return edge_index, edge_attr

def create_graph_label(solution):
    """Create graph-level label (makespan)"""
    return torch.tensor([solution['makespan']], dtype=torch.float32)

def solution_to_tensors(solution_path, app_path):
    """Convert a solution to PyTorch tensors"""
    solution = load_solution(solution_path)
    app_data = load_application(app_path)
    
    # Create tensors
    x = create_node_features(app_data, solution['schedule'])
    edge_index, edge_attr = create_edge_index(app_data)
    y = create_graph_label(solution)
    
    return {
        'x': x,  # Node features [num_nodes, num_features]
        'edge_index': edge_index,  # Edge connectivity [2, num_edges]
        'edge_attr': edge_attr,  # Edge features [num_edges, edge_features]
        'y': y,  # Graph label [1]
        'num_nodes': x.size(0),
        'num_edges': edge_index.size(1)
    }

def batch_solutions_to_tensors(solution_dir='solution', app_dir='Application', output_file='training_data.pt'):
    """Convert all valid solutions to tensors and save"""
    solution_files = glob.glob(f'{solution_dir}/*_ga.json')
    
    data_list = []
    valid_count = 0
    
    print(f"Found {len(solution_files)} solution files")
    
    for sol_path in solution_files:
        # Extract application name from solution filename
        # T2_var_001_seed00_ga.json -> T2_var_001.json
        filename = Path(sol_path).stem  # T2_var_001_seed00_ga or T2_var_001_ga
        if '_seed' in filename:
            app_name = filename.split('_seed')[0] + '.json'
        else:
            app_name = filename.replace('_ga', '') + '.json'
        
        app_path = f'{app_dir}/{app_name}'
        
        if not Path(app_path).exists():
            print(f"  Skipping {Path(sol_path).name}: Application {app_name} not found")
            continue
        
        try:
            tensors = solution_to_tensors(sol_path, app_path)
            data_list.append(tensors)
            valid_count += 1
            if valid_count <= 5:
                print(f"  + {Path(sol_path).name}: {tensors['num_nodes']} nodes, {tensors['num_edges']} edges, makespan={tensors['y'].item():.2f}")
        except Exception as e:
            print(f"  X Error processing {Path(sol_path).name}: {e}")
    
    if data_list:
        # Save as PyTorch tensors
        torch.save(data_list, output_file)
        print(f"\n+ Saved {len(data_list)} graphs to {output_file}")
        print(f"  Total nodes: {sum(d['num_nodes'] for d in data_list)}")
        print(f"  Total edges: {sum(d['num_edges'] for d in data_list)}")
        print(f"  Average makespan: {np.mean([d['y'].item() for d in data_list]):.2f}")
        return data_list
    else:
        print("\nX No valid solutions to convert")
        return []

if __name__ == "__main__":
    print("Converting GA solutions to PyTorch tensors for GNN training...\n")
    data = batch_solutions_to_tensors()
    
    if data:
        print(f"\nExample tensor structure:")
        print(f"  x (node features): {data[0]['x'].shape}")
        print(f"  edge_index: {data[0]['edge_index'].shape}")
        print(f"  edge_attr: {data[0]['edge_attr'].shape}")
        print(f"  y (makespan): {data[0]['y'].shape}")
