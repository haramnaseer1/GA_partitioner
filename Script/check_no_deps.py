"""Check tasks with no dependencies to find pure execution time"""
import json
import sys
import os

# Add src to path for auxiliary_fun_GA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import auxiliary_fun_GA as af

# --- CRITICAL CONSTANT ---
REFERENCE_SPEED_HZ = 1e9 # 1 GHz
# -------------------------

try:
    app = json.load(open('Application/T2_var_001.json'))
    sol = json.load(open('solution/T2_var_001_ga.json'))
except FileNotFoundError:
    print("Error: Application/T2_var_001.json or solution/T2_var_001_ga.json not found.")
    sys.exit(1)

# Find tasks with no dependencies
tasks_no_deps = [t for t in sol if len(t['dependencies']) == 0]

print(f"Tasks with NO dependencies: {len(tasks_no_deps)}\n")

# Get the processor details from the solution/platform
# Note: Assuming all nodes in T2_var_001 solution run at the same speed for simplicity
# We will use the speed of the node Task 0 is on for verification
task0_node_id = [t for t in sol if t['task_id'] == 0][0]['node_id']
try:
    G, _, _, clk_speed_raw = af.load_graph_from_json()
    clk_speed_str = clk_speed_raw.get(task0_node_id, "1 GHz")
    clk_speed_hz = af.convert_clocking_speed_to_hz(clk_speed_str)
except Exception:
    clk_speed_hz = 1e9 # Default to 1 GHz

print(f"Verifying against node {task0_node_id} clock speed: {clk_speed_hz/1e6:.1f} MHz\n")

print(f"{'Task':<6} {'wcet':<8} {'proc_t':<10} {'Duration':<18} {'Expected':<18} {'Match?'}")
print("=" * 70)

for task_sol in tasks_no_deps[:10]:
    task_id = task_sol['task_id']
    # Match application job by ID, not index
    task_app = [j for j in app['application']['jobs'] if j['id'] == task_id][0] 
    
    wcet = task_app['wcet_fullspeed']
    proc_time = task_app['processing_times']
    
    duration = task_sol['end_time'] - task_sol['start_time']
    
    # CORRECTED CALCULATION
    if clk_speed_hz > 0:
        expected_duration = proc_time * (REFERENCE_SPEED_HZ / clk_speed_hz)
    else:
        expected_duration = proc_time
    
    match = "✓" if abs(duration - expected_duration) < 1e-4 else f"✗ (diff: {abs(duration - expected_duration):.2f})"
    
    print(f"{task_id:<6} {wcet:<8} {proc_time:<10} {duration:<18.4f} {expected_duration:<18.4f} {match}")