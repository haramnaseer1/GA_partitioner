import json

app = json.load(open('Application/T2_var_001.json'))

print("=== Original Application Tasks 0, 1, 2 ===")
for i in [0, 1, 2]:
    task = [j for j in app['application']['jobs'] if j['id'] == i]
    if task:
        print(f"Task {i}: processing={task[0]['processing_times']}, can_run_on={task[0]['can_run_on']}")
    else:
        print(f"Task {i}: NOT FOUND")

print("\n=== Messages between tasks 0, 1, 2 ===")
msgs = [m for m in app['application']['messages'] 
        if (m['sender'] in [0,1,2] and m['receiver'] in [0,1,2])]
for m in msgs:
    print(f"  Msg {m['id']}: {m['sender']}->{m['receiver']}, size={m['size']}")
    
if not msgs:
    print("  NO messages between these tasks!")
