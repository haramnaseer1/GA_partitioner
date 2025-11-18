import json

# Load application model
with open('Application/T2_var_001.json') as f:
    app = json.load(f)

# Load solution
with open('solution/T2_var_001_ga.json') as f:
    sol = json.load(f)

print("Application tasks (processing times):")
for job in app['application']['jobs']:
    print(f"  Task {job['id']}: processing_time = {job.get('processing_time', 'N/A')}")

print("\nSolution schedule:")
for t in sorted(sol, key=lambda x: (x['node_id'], x['start_time'])):
    duration = t['end_time'] - t['start_time']
    print(f"  Task {t['task_id']} on proc {t['node_id']}: start={t['start_time']:.6f}, end={t['end_time']:.6f}, duration={duration:.6f}")

# Check Task 3 specifically
task3_app = next((j for j in app['application']['jobs'] if j['id'] == 3), None)
task3_sol = next((t for t in sol if t['task_id'] == 3), None)

print(f"\nTask 3 analysis:")
print(f"  App processing_time: {task3_app.get('processing_time', 'N/A')}")
print(f"  Solution duration: {task3_sol['end_time'] - task3_sol['start_time']:.6f}")
print(f"  Solution start: {task3_sol['start_time']:.6f}")
print(f"  Solution end: {task3_sol['end_time']:.6f}")
