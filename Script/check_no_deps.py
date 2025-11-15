"""Check tasks with no dependencies to find pure execution time"""
import json

app = json.load(open('Application/T2_var_001.json'))
sol = json.load(open('solution/T2_var_001_ga.json'))

# Find tasks with no dependencies
tasks_no_deps = [t for t in sol if len(t['dependencies']) == 0]

print(f"Tasks with NO dependencies: {len(tasks_no_deps)}\n")

clk_speed_hz = 333 * 1e6

print(f"{'Task':<6} {'wcet':<8} {'proc_t':<10} {'Duration (s)':<18} {'Implied Cycles':<15} {'proc_t/clock':<18}")
print("=" * 95)

for task_sol in tasks_no_deps[:10]:
    task_id = task_sol['task_id']
    task_app = app['application']['jobs'][task_id]
    
    wcet = task_app['wcet_fullspeed']
    proc_time = task_app['processing_times']
    
    duration = task_sol['end_time'] - task_sol['start_time']
    implied_cycles = duration * clk_speed_hz
    expected_from_proc_time = proc_time / clk_speed_hz
    
    print(f"{task_id:<6} {wcet:<8} {proc_time:<10} {duration:<18.10e} {implied_cycles:<15.2f} {expected_from_proc_time:<18.10e}")

# Check if wcet_fullspeed might be scaled
print("\n" + "=" * 95)
print("\nChecking if execution uses wcet_fullspeed instead:")
print(f"{'Task':<6} {'wcet':<8} {'Duration (s)':<18} {'wcet/clock':<18} {'Match?'}")
print("=" * 95)

for task_sol in tasks_no_deps[:10]:
    task_id = task_sol['task_id']
    task_app = app['application']['jobs'][task_id]
    
    wcet = task_app['wcet_fullspeed']
    duration = task_sol['end_time'] - task_sol['start_time']
    expected_from_wcet = wcet / clk_speed_hz
    
    match = "✓" if abs(duration - expected_from_wcet) < 1e-9 else f"✗ (diff: {abs(duration - expected_from_wcet):.2e})"
    
    print(f"{task_id:<6} {wcet:<8} {duration:<18.10e} {expected_from_wcet:<18.10e} {match}")
