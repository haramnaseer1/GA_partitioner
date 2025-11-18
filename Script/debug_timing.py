"""
Debug script to understand GA timing calculations
"""
import json
import sys
import os

# Add src to path for auxiliary_fun_GA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import auxiliary_fun_GA as af

# --- CRITICAL CONSTANT ---
REFERENCE_SPEED_HZ = 1e9 # 1 GHz
# -------------------------

# Load data (assuming T2_var_001.json is the target app)
try:
    app = json.load(open('Application/T2_var_001.json'))
    sol = json.load(open('solution/T2_var_001_ga.json'))
except FileNotFoundError:
    print("Error: Application/T2_var_001.json or solution/T2_var_001_ga.json not found.")
    sys.exit(1)

# Get the processor details from the solution/platform
# Note: This debug script assumes all tasks are on the same node for simplicity, 
# but we will try to find the node Task 0 is on.
task0_node_id = [t for t in sol if t['task_id'] == 0][0]['node_id']

try:
    # We need to load the platform to get the real clock speed
    G, _, _, clk_speed_raw = af.load_graph_from_json()
    clk_speed_str = clk_speed_raw.get(task0_node_id, "1 GHz")
    clk_speed_hz = af.convert_clocking_speed_to_hz(clk_speed_str)
except Exception as e:
    print(f"Error loading platform data or node clock speed: {e}. Defaulting to 1 GHz.")
    clk_speed_hz = 1e9
    clk_speed_str = "1 GHz"

clk_speed_mhz = clk_speed_hz / 1e6

print("=" * 80)
print("TIMING ANALYSIS FOR T2_var_001")
print("=" * 80)
print(f"\nTask 0 Node: {task0_node_id} | Clock Speed: {clk_speed_mhz:.1f} MHz ({clk_speed_str})")
print(f"\n{'Task':<6} {'wcet':<8} {'proc_t':<10} {'Actual Duration':<20} {'Expected Duration':<20} {'Formula Check'}")
print("=" * 100)

for i in range(10):  # First 10 tasks
    task_app = [j for j in app['application']['jobs'] if j['id'] == i][0]
    task_sol = [t for t in sol if t['task_id'] == i]
    if not task_sol: continue
    task_sol = task_sol[0]
    
    task_id = task_app['id']
    wcet = task_app['wcet_fullspeed']
    proc_time = task_app['processing_times']
    
    start = task_sol['start_time']
    end = task_sol['end_time']
    actual_duration = end - start
    
    # CORRECTED TIMING CALCULATION
    if clk_speed_hz > 0:
        expected_duration_from_proc_time = proc_time * (REFERENCE_SPEED_HZ / clk_speed_hz)
    else:
        expected_duration_from_proc_time = proc_time
    
    
    # Check which formula matches
    if abs(actual_duration - expected_duration_from_proc_time) < 1e-4:
        formula = "WCET/Scaled Clock ✓"
    else:
        formula = f"✗ (Diff: {abs(actual_duration - expected_duration_from_proc_time):.2f})"
    
    print(f"{task_id:<6} {wcet:<8} {proc_time:<10} {actual_duration:<20.4f} {expected_duration_from_proc_time:<20.4f} {formula}")