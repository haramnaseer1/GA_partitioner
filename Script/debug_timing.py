"""
Debug script to understand GA timing calculations
"""
import json

# Load data
app = json.load(open('Application/T2_var_001.json'))
sol = json.load(open('solution/T2_var_001_ga.json'))

# Platform clock speed
clk_speed_mhz = 333
clk_speed_hz = clk_speed_mhz * 1e6

print("=" * 80)
print("TIMING ANALYSIS FOR T2_var_001")
print("=" * 80)
print(f"\nPlatform Clock Speed: {clk_speed_mhz} MHz = {clk_speed_hz:,.0f} Hz")
print("\n" + "=" * 80)
print(f"{'Task':<6} {'wcet':<8} {'proc_t':<10} {'Actual Duration (s)':<20} {'Implied Cycles':<15} {'Formula Check'}")
print("=" * 80)

for i in range(10):  # First 10 tasks
    task_app = app['application']['jobs'][i]
    task_sol = [t for t in sol if t['task_id'] == i][0]
    
    task_id = task_app['id']
    wcet = task_app['wcet_fullspeed']
    proc_time = task_app['processing_times']
    
    start = task_sol['start_time']
    end = task_sol['end_time']
    actual_duration = end - start
    
    # What the GA formula says it should be
    expected_duration_from_proc_time = proc_time / clk_speed_hz
    expected_duration_from_wcet = wcet / clk_speed_hz
    
    # Work backwards from actual
    implied_cycles = actual_duration * clk_speed_hz
    
    # Check which formula matches
    if abs(actual_duration - expected_duration_from_proc_time) < 1e-9:
        formula = "proc_time/clock ✓"
    elif abs(actual_duration - expected_duration_from_wcet) < 1e-9:
        formula = "wcet/clock ✓"
    else:
        formula = f"UNKNOWN (error: {abs(actual_duration - expected_duration_from_proc_time):.2e})"
    
    print(f"{task_id:<6} {wcet:<8} {proc_time:<10} {actual_duration:<20.10e} {implied_cycles:<15.2f} {formula}")

print("=" * 80)
print("\nSUMMARY:")
print(f"If GA uses: execution_time = processing_time / clock_speed")
print(f"  → Task 0 would take: {app['application']['jobs'][0]['processing_times'] / clk_speed_hz:.10e} seconds")
print(f"  → Actual Task 0 duration: {[t for t in sol if t['task_id']==0][0]['end_time']:.10e} seconds")
print(f"  → Match: {abs(app['application']['jobs'][0]['processing_times'] / clk_speed_hz - [t for t in sol if t['task_id']==0][0]['end_time']) < 1e-9}")
print("\n" + "=" * 80)
