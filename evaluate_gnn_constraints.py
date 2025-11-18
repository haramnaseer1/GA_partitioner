"""
Evaluate GNN with Constraint Enforcement
=========================================

Tests the GNN model with constrained inference and validates against all constraints:
1. Eligibility
2. Precedence 
3. Non-overlap
4. Timing consistency

Compares GNN+constraints vs GA solutions
"""

import torch
import json
import sys
from pathlib import Path
from constrained_inference import ConstrainedScheduler
from train_gnn_multitask import create_multitask_model
from torch_geometric.data import Data
import subprocess


def load_application(app_path):
    """Load application JSON"""
    with open(app_path, 'r') as f:
        return json.load(f)


def create_graph_from_app(app_data):
    """Create PyG Data object from application"""
    jobs = app_data['application']['jobs']
    messages = app_data['application']['messages']
    
    num_tasks = len(jobs)
    
    # Node features: [processing_time, can_run_on_count, num_deps]
    node_features = []
    for job in jobs:
        processing_time = job.get('processing_times', 100)
        can_run_on = job.get('can_run_on', [1])
        
        # Count dependencies
        num_deps = sum(1 for msg in messages if msg['receiver'] == job['id'])
        
        node_features.append([
            processing_time,
            len(can_run_on),
            num_deps
        ])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge index and attributes from messages
    edge_index = []
    edge_attr = []
    for msg in messages:
        edge_index.append([msg['sender'], msg['receiver']])
        edge_attr.append([msg.get('size', 24)])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(num_tasks, dtype=torch.long)
    )
    
    return data


def validate_solution(solution, app_data, platform_path='Path_Information/3_Tier_Platform.json'):
    """
    Validate solution against all constraints
    
    Returns:
        dict: Validation results with pass/fail for each constraint
    """
    scheduler = ConstrainedScheduler(platform_path)
    jobs = app_data['application']['jobs']
    messages = app_data['application']['messages']
    
    results = {
        'eligibility': {'pass': True, 'violations': []},
        'precedence': {'pass': True, 'violations': []},
        'non_overlap': {'pass': True, 'violations': []},
        'timing': {'pass': True, 'violations': []}
    }
    
    # Build solution map
    solution_map = {s['task_id']: s for s in solution}
    
    # 1. ELIGIBILITY CHECK
    for job in jobs:
        task_id = job['id']
        if task_id not in solution_map:
            continue
        
        assigned_node = solution_map[task_id]['node_id']
        can_run_on = job.get('can_run_on', [1])
        
        # Check if assigned node type matches can_run_on
        node_type = scheduler.processor_types.get(assigned_node, 'Unknown')
        allowed_types = [scheduler.resource_mapping[rt] for rt in can_run_on]
        
        if node_type not in allowed_types:
            results['eligibility']['pass'] = False
            results['eligibility']['violations'].append(
                f"Task {task_id} assigned to {node_type} (node {assigned_node}), "
                f"but can only run on {allowed_types}"
            )
    
    # 2. PRECEDENCE CHECK
    for msg in messages:
        sender = msg['sender']
        receiver = msg['receiver']
        
        if sender not in solution_map or receiver not in solution_map:
            continue
        
        sender_end = solution_map[sender]['end_time']
        receiver_start = solution_map[receiver]['start_time']
        
        # Compute required delay
        sender_node = solution_map[sender]['node_id']
        receiver_node = solution_map[receiver]['node_id']
        msg_size = msg.get('size', 24)
        
        comm_delay = scheduler._compute_comm_delay(sender_node, receiver_node, msg_size)
        required_start = sender_end + comm_delay
        
        if receiver_start < required_start - 1e-6:  # Small epsilon for floating point
            results['precedence']['pass'] = False
            results['precedence']['violations'].append(
                f"Task {sender} -> {receiver}: Required start {required_start:.4f}, "
                f"Actual start {receiver_start:.4f}, Gap {receiver_start - sender_end:.4f}"
            )
    
    # 3. NON-OVERLAP CHECK (per processor)
    tasks_by_processor = {}
    for s in solution:
        node_id = s['node_id']
        if node_id not in tasks_by_processor:
            tasks_by_processor[node_id] = []
        tasks_by_processor[node_id].append(s)
    
    for node_id, tasks in tasks_by_processor.items():
        if len(tasks) <= 1:
            continue
        
        # Sort by start time
        tasks_sorted = sorted(tasks, key=lambda t: t['start_time'])
        
        for i in range(len(tasks_sorted) - 1):
            curr = tasks_sorted[i]
            next_task = tasks_sorted[i + 1]
            
            if curr['end_time'] > next_task['start_time'] + 1e-6:
                results['non_overlap']['pass'] = False
                results['non_overlap']['violations'].append(
                    f"Node {node_id}: Task {curr['task_id']} ends at {curr['end_time']:.4f}, "
                    f"but Task {next_task['task_id']} starts at {next_task['start_time']:.4f}"
                )
    
    # 4. TIMING CONSISTENCY
    for job in jobs:
        task_id = job['id']
        if task_id not in solution_map:
            continue
        
        s = solution_map[task_id]
        duration = s['end_time'] - s['start_time']
        
        # Compute expected duration
        proc_id = s['node_id']
        processing_time = job.get('processing_times', 100)
        clock_speed = scheduler.processor_speeds.get(proc_id, 1e9)
        expected_duration = processing_time / clock_speed
        
        # Allow small tolerance
        if abs(duration - expected_duration) > 1e-3:
            results['timing']['pass'] = False
            results['timing']['violations'].append(
                f"Task {task_id}: Duration {duration:.6f}, Expected {expected_duration:.6f}"
            )
    
    return results


def evaluate_app(app_name, model, scheduler):
    """Evaluate GNN on a single application"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {app_name}")
    print(f"{'='*70}")
    
    app_path = f'Application/{app_name}.json'
    ga_solution_path = f'solution/{app_name}_ga.json'
    
    # Load application
    if not Path(app_path).exists():
        print(f"❌ Application not found: {app_path}")
        return None
    
    app_data = load_application(app_path)
    
    # Create graph
    data = create_graph_from_app(app_data)
    
    # Run GNN with constraints
    print("Running GNN with constraint enforcement...")
    constrained = scheduler.predict_with_constraints(model, data, app_path)
    gnn_solution = scheduler.to_solution_format(constrained, app_data)
    
    # Get GNN makespan
    gnn_makespan = constrained['makespan'].item()
    print(f"✓ GNN Makespan: {gnn_makespan:.2f}")
    
    # Validate GNN solution
    print("\nValidating GNN solution...")
    gnn_validation = validate_solution(gnn_solution, app_data)
    
    all_pass = all(v['pass'] for v in gnn_validation.values())
    if all_pass:
        print("✓✓✓ ALL CONSTRAINTS SATISFIED!")
    else:
        print("❌ CONSTRAINT VIOLATIONS FOUND:")
        for constraint, result in gnn_validation.items():
            if not result['pass']:
                print(f"\n  {constraint.upper()}:")
                for violation in result['violations'][:3]:  # Show first 3
                    print(f"    - {violation}")
                if len(result['violations']) > 3:
                    print(f"    ... and {len(result['violations']) - 3} more")
    
    # Compare with GA if available
    ga_makespan = None
    if Path(ga_solution_path).exists():
        with open(ga_solution_path, 'r') as f:
            ga_solution = json.load(f)
        ga_makespan = max(t['end_time'] for t in ga_solution)
        
        print(f"\nGA Makespan: {ga_makespan:.2f}")
        delta = gnn_makespan - ga_makespan
        delta_pct = (delta / ga_makespan) * 100
        print(f"Difference: {delta:+.2f} ({delta_pct:+.2f}%)")
        
        if delta < 0:
            print("🎉 GNN IMPROVED over GA!")
        elif abs(delta_pct) < 5:
            print("✓ Within 5% of GA")
        elif abs(delta_pct) < 10:
            print("○ Within 10% of GA")
        else:
            print("⚠ More than 10% difference")
    
    return {
        'app_name': app_name,
        'gnn_makespan': gnn_makespan,
        'ga_makespan': ga_makespan,
        'constraints_pass': all_pass,
        'validation': gnn_validation
    }


def main():
    """Main evaluation"""
    print("="*70)
    print("GNN EVALUATION WITH CONSTRAINT ENFORCEMENT")
    print("="*70)
    
    # Load model
    print("\nLoading GNN model...")
    model = create_multitask_model()
    
    try:
        checkpoint = torch.load('models_multitask/best_model.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained model")
    except Exception as e:
        print(f"⚠ Could not load trained model: {e}")
        print("Using untrained model for testing...")
    
    # Initialize scheduler
    scheduler = ConstrainedScheduler()
    
    # Test applications
    test_apps = sys.argv[1:] if len(sys.argv) > 1 else ['T2', 'T20']
    
    results = []
    for app_name in test_apps:
        result = evaluate_app(app_name, model, scheduler)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if results:
        passed = sum(1 for r in results if r['constraints_pass'])
        print(f"\nConstraint Satisfaction: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
        
        if any(r['ga_makespan'] for r in results):
            deltas = [
                (r['gnn_makespan'] - r['ga_makespan']) / r['ga_makespan'] * 100
                for r in results if r['ga_makespan']
            ]
            if deltas:
                print(f"Makespan vs GA:")
                print(f"  Mean: {sum(deltas)/len(deltas):+.2f}%")
                print(f"  Median: {sorted(deltas)[len(deltas)//2]:+.2f}%")
                print(f"  Improvements: {sum(1 for d in deltas if d < 0)}/{len(deltas)}")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
