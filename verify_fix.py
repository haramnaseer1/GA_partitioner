import json

with open('solution/T2_var_003_ga.json') as f:
    sol = json.load(f)
    proc_ids = sorted(set(t['node_id'] for t in sol))
    
print(f"Solution processor IDs: {proc_ids}")
print(f"Expected Platform 2 range: [21-26]")

if set(proc_ids).issubset(set(range(21, 27))):
    print("✓ MATCH! Fix successful - solution uses Platform 2 processors!")
else:
    print("✗ MISMATCH - solution still uses wrong platform")
