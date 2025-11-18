import json

with open('solution/T2_var_001_ga.json') as f:
    sol = json.load(f)

print("Full schedule:")
for t in sorted(sol, key=lambda x: (x['node_id'], x['start_time'], x['task_id'])):
    print(f"  Task {t['task_id']:3d} on proc {t['node_id']:2d}: start={t['start_time']:8.2f}, end={t['end_time']:8.2f}")

print("\nChecking for exact duplicates (same start AND end on same processor):")
from collections import defaultdict
by_proc = defaultdict(list)
for t in sol:
    by_proc[t['node_id']].append(t)

for proc_id, tasks in sorted(by_proc.items()):
    time_slots = defaultdict(list)
    for t in tasks:
        key = (t['start_time'], t['end_time'])
        time_slots[key].append(t['task_id'])
    
    for (start, end), task_ids in time_slots.items():
        if len(task_ids) > 1:
            print(f"  Processor {proc_id}: {len(task_ids)} tasks at EXACT same time {start:.2f}-{end:.2f}: tasks {task_ids}")
