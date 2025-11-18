import json

# Check T2_var_003 solution
with open('solution/T2_var_003_ga.json') as f:
    sol = json.load(f)
    sol_proc_ids = sorted(set(t['node_id'] for t in sol))
    print(f"T2_var_003 solution processor IDs: {sol_proc_ids}")

# Check Platform 2
with open('Platform/2_Platform.json') as f:
    plat = json.load(f)
    plat_proc_ids = sorted([n['id'] for n in plat['platform']['nodes'] if not n.get('is_router', False)])
    print(f"Platform 2 processor IDs: {plat_proc_ids}")

# Check EdgeAI-Trust platform
with open('Platform/EdgeAI-Trust_Platform.json') as f:
    edge_plat = json.load(f)
    edge_proc_ids = sorted([n['id'] for n in edge_plat['platform']['nodes'] if not n.get('is_router', False)])
    print(f"EdgeAI-Trust processor IDs: {edge_proc_ids}")

print("\nMismatch detected!" if set(sol_proc_ids) != set(plat_proc_ids) else "\nMatch found!")
