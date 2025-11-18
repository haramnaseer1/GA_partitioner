"""
Convert GA solutions to PyTorch tensors for Multi-Task GNN training
Includes node-level labels for:
  - Processor assignment (classification)
  - Start/end times (regression)
  - Makespan (graph-level regression)
"""

import json
import torch
import numpy as np
from pathlib import Path
import glob
from torch_geometric.data import Data

def load_solution(solution_path, min_tasks=3):
    """Load a solution JSON file
    Args:
        solution_path: Path to solution JSON  
        min_tasks: Minimum tasks required (default: 3)
    """
    with open(solution_path, 'r') as f:
        data = json.load(f)
        
        # Skip empty or too-small solutions
        if not data or len(data) < min_tasks:
            return None
        
        # Keep full task data with all fields
        tasks_by_id = {task['task_id']: task for task in data}
        makespan = max(task['end_time'] for task in data)
        return {'tasks': tasks_by_id, 'makespan': makespan}

def load_application(app_path):
    """Load application JSON file"""
    with open(app_path, 'r') as f:
        return json.load(f)['application']

def load_platform(platform_path='Path_Information/3_Tier_Platform.json'):
    """Load platform model to get processor list"""
    with open(platform_path, 'r') as f:
        platform = json.load(f)
        # Extract compute nodes (non-routers)
        compute_nodes = [node for node in platform['platform']['nodes'] if not node.get('is_router', False)]
        # Create processor ID mapping
        processor_ids = sorted([node['id'] for node in compute_nodes])
        processor_to_idx = {pid: idx for idx, pid in enumerate(processor_ids)}
        return {
            'processor_ids': processor_ids,
            'processor_to_idx': processor_to_idx,
            'num_processors': len(processor_ids)
        }

def create_node_features(app_data):
    """Create node feature matrix for tasks (WITHOUT schedule info)
    Features: [processing_time, deadline, num_dependencies]
    This represents the INPUT to the GNN (task graph only)
    """
    jobs = {job['id']: job for job in app_data['jobs']}
    messages = app_data['messages']
    
    # Count incoming messages (dependencies) per task
    dep_count = {job_id: 0 for job_id in jobs.keys()}
    for msg in messages:
        dep_count[msg['receiver']] = dep_count.get(msg['receiver'], 0) + 1
    
    features = []
    for task_id in sorted(jobs.keys()):
        job = jobs[task_id]
        features.append([
            float(job['processing_times']),
            float(job['deadline']),
            float(dep_count.get(task_id, 0))
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

def create_node_labels(solution_tasks, app_data, platform_info):
    """Create node-level labels from GA solution
    Returns:
        - processor_labels: processor index for each task [num_nodes]
        - start_labels: start time for each task [num_nodes]
        - end_labels: end time for each task [num_nodes]
    """
    jobs = {job['id']: job for job in app_data['jobs']}
    num_tasks = len(jobs)
    
    # Convert solution tasks list to dict for easy lookup
    tasks_dict = {task['task_id']: task for task in solution_tasks}
    
    # Skip if tasks don't match (incomplete/corrupt solution)
    missing_tasks = set(jobs.keys()) - set(tasks_dict.keys())
    if missing_tasks:
        return None  # Skip this solution
    
    processor_labels = []
    start_labels = []
    end_labels = []
    
    for task_id in sorted(jobs.keys()):
        task = tasks_dict[task_id]
        
        # Processor assignment (convert to class index)
        processor_id = task['node_id']
        processor_idx = platform_info['processor_to_idx'].get(processor_id, 0)
        processor_labels.append(processor_idx)
        
        # Timing
        start_labels.append(float(task['start_time']))
        end_labels.append(float(task['end_time']))
    
    return (
        torch.tensor(processor_labels, dtype=torch.long),  # Classification target
        torch.tensor(start_labels, dtype=torch.float32),   # Regression target
        torch.tensor(end_labels, dtype=torch.float32)      # Regression target
    )

def solution_to_graph_data(solution_path, app_path, platform_info, min_tasks=3):
    """Convert a solution to PyTorch Geometric Data object with multi-task labels
    Args:
        min_tasks: Minimum tasks required (passed to load_solution)
    """
    solution = load_solution(solution_path, min_tasks=min_tasks)
    if solution is None:
        return None  # Skip empty/invalid solutions
    
    app_data = load_application(app_path)
    
    # Input features (task graph only - no solution info)
    x = create_node_features(app_data)
    edge_index, edge_attr = create_edge_index(app_data)
    
    # Node-level labels (targets)
    labels = create_node_labels(
        solution['tasks'].values(), app_data, platform_info
    )
    if labels is None:
        return None  # Skip if label creation failed
    
    processor_labels, start_labels, end_labels = labels
    
    # Graph-level label
    makespan = torch.tensor([solution['makespan']], dtype=torch.float32)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,  # [num_nodes, 3] - INPUT features (processing_time, deadline, dependencies)
        edge_index=edge_index,  # [2, num_edges]
        edge_attr=edge_attr,  # [num_edges, 1]
        
        # Node-level targets (what GNN should predict per task)
        y_processor=processor_labels,  # [num_nodes] - processor assignment
        y_start=start_labels,          # [num_nodes] - start time
        y_end=end_labels,              # [num_nodes] - end time
        
        # Graph-level target
        y_makespan=makespan,           # [1] - overall makespan
        
        # Metadata
        num_nodes=x.size(0),
        num_processors=platform_info['num_processors']
    )
    
    return data

def batch_solutions_to_tensors(
    solution_dir='solution', 
    app_dir='Application', 
    output_file='training_data_multitask.pt',
    platform_path='Path_Information/3_Tier_Platform.json',
    min_tasks=3
):
    """Convert all valid solutions to multi-task training data
    Args:
        min_tasks: Minimum tasks per solution to include
    """
    
    # Load platform info (shared across all solutions)
    print("Loading platform information...")
    platform_info = load_platform(platform_path)
    print(f"  Found {platform_info['num_processors']} compute processors\n")
    
    solution_files = glob.glob(f'{solution_dir}/*_ga.json')
    
    data_list = []
    valid_count = 0
    skipped_count = 0
    
    print(f"Converting {len(solution_files)} solution files (min_tasks={min_tasks})...\n")
    
    for sol_path in solution_files:
        # Extract application name from solution filename
        filename = Path(sol_path).stem
        if '_seed' in filename:
            app_name = filename.split('_seed')[0] + '.json'
        else:
            app_name = filename.replace('_ga', '') + '.json'
        
        app_path = f'{app_dir}/{app_name}'
        
        if not Path(app_path).exists():
            skipped_count += 1
            continue
        
        try:
            data = solution_to_graph_data(sol_path, app_path, platform_info, min_tasks=min_tasks)
            
            # Skip invalid/incomplete solutions
            if data is None:
                skipped_count += 1
                continue
            
            data_list.append(data)
            valid_count += 1
            
            if valid_count <= 5:
                print(f"  [OK] {Path(sol_path).name}:")
                print(f"      {data.num_nodes} tasks, {data.edge_index.size(1)} dependencies")
                print(f"      Makespan: {data.y_makespan.item():.2f}")
                print(f"      Processors used: {torch.unique(data.y_processor).tolist()}")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {Path(sol_path).name}:")
            print(f"     {type(e).__name__}: {e}")
            if valid_count < 2:  # Print traceback for first few errors
                traceback.print_exc()
    
    if data_list:
        # Save as PyTorch Geometric data list
        torch.save(data_list, output_file)
        
        print(f"\n{'='*70}")
        print(f"DATASET CREATED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Saved {len(data_list)} graphs to {output_file}")
        print(f"Skipped {skipped_count} invalid/small solutions")
        print(f"\nDataset Statistics:")
        print(f"  Total tasks: {sum(d.num_nodes for d in data_list)}")
        print(f"  Total dependencies: {sum(d.edge_index.size(1) for d in data_list)}")
        print(f"  Avg tasks per graph: {np.mean([d.num_nodes for d in data_list]):.1f}")
        print(f"  Makespan range: {min(d.y_makespan.item() for d in data_list):.1f} - {max(d.y_makespan.item() for d in data_list):.1f}")
        print(f"  Num processors: {platform_info['num_processors']}")
        print(f"\nLabel Information:")
        print(f"  Node-level labels per task:")
        print(f"    • y_processor: Processor assignment (0-{platform_info['num_processors']-1})")
        print(f"    • y_start: Start time (float)")
        print(f"    • y_end: End time (float)")
        print(f"  Graph-level label:")
        print(f"    • y_makespan: Overall makespan (float)")
        print(f"{'='*70}\n")
        
        return data_list
    else:
        print("\n[ERROR] No valid solutions to convert")
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert GA solutions to multi-task training tensors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_tensors_multitask.py
  python create_tensors_multitask.py --output data/training_multitask.pt
  python create_tensors_multitask.py --solution-dir solution --app-dir Application
        """
    )
    
    parser.add_argument('--solution-dir', type=str, default='solution',
                        help='Directory containing GA solution JSON files (default: solution)')
    parser.add_argument('--app-dir', type=str, default='Application',
                        help='Directory containing application JSON files (default: Application)')
    parser.add_argument('--output', '-o', type=str, default='training_data_multitask.pt',
                        help='Output file path for training tensors (default: training_data_multitask.pt)')
    parser.add_argument('--platform', type=str, default='Path_Information/3_Tier_Platform.json',
                        help='Platform JSON file path (default: Path_Information/3_Tier_Platform.json)')
    parser.add_argument('--min-tasks', type=int, default=3,
                        help='Minimum tasks per solution to include (default: 3)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MULTI-TASK GNN TRAINING DATA GENERATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Solution dir: {args.solution_dir}")
    print(f"  Application dir: {args.app_dir}")
    print(f"  Output file: {args.output}")
    print(f"  Platform file: {args.platform}")
    print(f"  Min tasks filter: {args.min_tasks}")
    print("\nConverting GA solutions to multi-task training format...\n")
    
    data = batch_solutions_to_tensors(
        solution_dir=args.solution_dir,
        app_dir=args.app_dir,
        output_file=args.output,
        platform_path=args.platform,
        min_tasks=args.min_tasks
    )
    
    if data:
        print("\nExample data structure:")
        sample = data[0]
        print(f"  Input features (x): {sample.x.shape} - [num_tasks, 3]")
        print(f"  Edge index: {sample.edge_index.shape}")
        print(f"  Edge attributes: {sample.edge_attr.shape}")
        print(f"\n  Target labels:")
        print(f"    y_processor: {sample.y_processor.shape} - processor per task")
        print(f"    y_start: {sample.y_start.shape} - start time per task")
        print(f"    y_end: {sample.y_end.shape} - end time per task")
        print(f"    y_makespan: {sample.y_makespan.shape} - graph makespan")
        
        print(f"\n[SUCCESS] Ready for multi-task GNN training!")
        print(f"[SUCCESS] Training data saved to: {args.output}")
    else:
        import sys
        sys.exit(1)
