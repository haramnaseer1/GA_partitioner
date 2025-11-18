"""Extract the actual clocking_speed dict from GA's load function"""
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Now we can import
import auxiliary_fun_GA as af
import reading_application_model as ra

# --- CRITICAL CONSTANT ---
# Reference speed used for WCET normalization (1 GHz = 1e9 Hz)
REFERENCE_SPEED_HZ = 1e9 
# -------------------------

# Load the platform
print("Loading platform data...")
G, prcr_list, unique_edge_pcr, clk_speed_raw = af.load_graph_from_json()

# ... (rest of clock loading remains the same)

print(f"\nRaw clocking speeds from platform:")
for proc_id, speed_str in clk_speed_raw.items():
    print(f"  Processor {proc_id}: {speed_str}")

print(f"\nConverting to Hz...")
clocking_speed_hz = {processor_id: af.convert_clocking_speed_to_hz(speed) 
                      for processor_id, speed in clk_speed_raw.items()}

for proc_id, speed_hz in clocking_speed_hz.items():
    print(f"  Processor {proc_id}: {speed_hz:,.0f} Hz = {speed_hz/1e6:.1f} MHz")

# Now check what the actual task duration formula gives us
print(f"\n{'='*80}")
print("TESTING FORMULA: execution_time = WCET * (Reference_Speed / Node_Speed)")
print(f"Reference Speed: {REFERENCE_SPEED_HZ:,.0f} Hz")
print(f"{'='*80}")

AM = json.load(open('Application/T2_var_001.json'))
proc_times = ra.find_processing_time(AM)

# Assume all tasks run on processor 51 (first CPU/FPGA in Edge Tier)
test_proc = 51
if test_proc not in clocking_speed_hz:
    test_proc = list(clocking_speed_hz.keys())[0] # Fallback to first found processor
    
clock_hz = clocking_speed_hz[test_proc]

print(f"\nAssuming tasks run on processor {test_proc} with clock {clock_hz:,.0f} Hz:")
print(f"\n{'Task':<6} {'proc_time':<12} {'calc_duration':<18}")
print("=" * 40)
for task_id in range(5):
    proc_time = proc_times[task_id]
    
    # CORRECTED CALCULATION
    if clock_hz > 0:
        calc_duration = proc_time * (REFERENCE_SPEED_HZ / clock_hz)
    else:
        calc_duration = proc_time # Default to proc_time if clock is 0
        
    print(f"{task_id:<6} {proc_time:<12} {calc_duration:<18.4f}")
    
# Compare with actual solution (This part relies on a valid solution being present)
sol = json.load(open('solution/T2_var_001_ga.json'))
print(f"\n{'='*80}")
print("ACTUAL DURATIONS FROM SOLUTION (for comparison):")
print(f"{'='*80}")
print(f"\n{'Task':<6} {'Node':<6} {'Actual Duration':<18}")
print("=" * 35)
for task_sol in [t for t in sol if t['task_id'] < 5]:
    task_id = task_sol['task_id']
    node_id = task_sol['node_id']
    duration = task_sol['end_time'] - task_sol['start_time']
    print(f"{task_id:<6} {node_id:<6} {duration:<18.4f}")