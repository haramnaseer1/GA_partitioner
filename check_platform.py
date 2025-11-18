import json

with open('solution/T20_ga.json') as f:
    data = json.load(f)

node_ids = sorted(set(t['node_id'] for t in data))
print(f"Unique node_ids in T20 solution: {node_ids}")
print(f"Min node_id: {min(node_ids)}, Max node_id: {max(node_ids)}")

# Determine platform based on node ID ranges
if any(51 <= n <= 59 for n in node_ids):
    print("Platform: 5 (nodes 51-59)")
elif any(31 <= n <= 39 for n in node_ids):
    print("Platform: 3 (nodes 31-39)")
elif any(21 <= n <= 29 for n in node_ids):
    print("Platform: 2 (nodes 21-29)")
