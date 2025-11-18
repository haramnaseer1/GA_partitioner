"""Detailed timing analysis for specific solutions"""
import json

def analyze_solution(sol_path, app_path):
    """Detailed analysis of a single solution"""
    
    with open(app_path, 'r') as f:
        app = json.load(f)
    
    with open(sol_path, 'r') as f:
        solution = json.load(f)
    
    # Build lookup tables
    task_times = {job['id']: job['processing_times'] for job in app['application']['jobs']}
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {sol_path.split('/')[-1]}")
    print(f"{'='*70}")
    
    for task in solution:
        tid = task['task_id']
        node = task['node_id']
        start = task['start_time']
        end = task['end_time']
        duration = end - start
        expected_proc = task_times.get(tid, 'N/A')
        
        print(f"\nTask {tid} (node {node}):")
        print(f"  Start: {start}")
        print(f"  End: {end}")
        print(f"  Duration: {duration:.12f}")
        print(f"  Expected processing_time: {expected_proc}")
        
        # Check if divided by clock speed
        if expected_proc != 'N/A':
            ratio = duration / expected_proc if expected_proc > 0 else 0
            if ratio > 0.9:  # Close to 1 = NOT divided
                print(f"  *** ISSUE: Duration matches processing_time (ratio={ratio:.4f})")
                print(f"  *** Processing time NOT divided by clock speed!")
            elif ratio < 0.00001:  # Very small = correctly divided
                print(f"  OK: Duration is very small (ratio={ratio:.2e}), likely divided by clock speed")
            else:
                print(f"  Ratio: {ratio:.2e}")
        
        # Check dependencies
        if task['dependencies']:
            print(f"  Dependencies:")
            for dep in task['dependencies']:
                print(f"    Task {dep['task_id']} -> path={dep['path_id']}, msg_size={dep['message_size']}")

# Analyze specific files
files_to_check = [
    ('solution/T2_ga.json', 'Application/T2.json'),
    ('solution/T20_ga.json', 'Application/T20.json'),
    ('solution/T2_var_001_ga.json', 'Application/T2_var_001.json'),
    ('solution/T2_var_002_ga.json', 'Application/T2_var_002.json'),
]

for sol, app in files_to_check:
    try:
        analyze_solution(sol, app)
    except FileNotFoundError:
        print(f"\nSkipping {sol} (not found)")
    except Exception as e:
        print(f"\nError analyzing {sol}: {e}")

print(f"\n{'='*70}")
print("Analysis complete")
print(f"{'='*70}")
