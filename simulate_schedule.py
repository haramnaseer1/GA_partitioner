"""
Simulate the scheduling logic to reproduce the overlap bug.
"""

# Simulated data from T2_var_001
processor = 26
task1_start = 374.0
task1_duration = 300.0
task1_end = 674.0

task3_deps_ready = 398.000001  # Based on dependencies

# The scheduling logic should do:
current_time_per_processor = {26: 0}

# Schedule Task 1
start_time = max(task1_start, current_time_per_processor[26])
end_time = start_time + task1_duration
print(f"Task 1: start={start_time}, end={end_time}")
current_time_per_processor[26] = end_time
print(f"Processor 26 busy until: {current_time_per_processor[26]}")

# Schedule Task 3
# Bug hypothesis: current_time_per_processor[26] not being checked properly
start_time_task3 = max(task3_deps_ready, current_time_per_processor[26])
print(f"\nTask 3 dependencies ready at: {task3_deps_ready}")
print(f"Processor 26 available at: {current_time_per_processor[26]}")
print(f"Task 3 SHOULD start at: {start_time_task3}")
print(f"Task 3 ACTUALLY started at: 398.000001")

if start_time_task3 != 398.000001:
    print(f"\n✗ BUG DETECTED: Task 3 started too early!")
    print(f"  Should start: {start_time_task3}")
    print(f"  Actually started: 398.000001")
    print(f"  This creates overlap from {398.000001} to {task1_end}")
