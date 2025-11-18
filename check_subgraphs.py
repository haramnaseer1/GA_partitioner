import json

# Check what's in SubGraphs files
with open('SubGraphs/1.json') as f:
    sg1 = json.load(f)
    
print("SubGraphs/1.json:")
print(f"  Tasks: {[j['id'] for j in sg1['application']['jobs']]}")
print(f"  Messages: {[(m['id'], m['sender'], m['receiver'], m['size']) for m in sg1['application']['messages']]}")

# Check Combined
with open('Combined_SubGraphs/1.json') as f:
    csg1 = json.load(f)
    
print("\nCombined_SubGraphs/1.json:")
print(f"  Tasks: {[j['id'] for j in csg1['application']['jobs']]}")
print(f"  Messages: {[(m['id'], m['sender'], m['receiver'], m['size']) for m in csg1['application']['messages']]}")

# Check which SubGraphs files are combined
with open('SubGP_from_DFS_/1.txt') as f:
    partitions = [int(p.strip()) for p in f.readlines()]
    
print(f"\nPartitions in layer 1: {partitions}")

# Check each partition
for p in partitions:
    with open(f'SubGraphs/{p}.json') as f:
        sg = json.load(f)
    print(f"\nSubGraphs/{p}.json:")
    print(f"  Tasks: {[j['id'] for j in sg['application']['jobs']]}")
    print(f"  Messages: {[(m.get('id', 'N/A'), m['sender'], m['receiver'], m['size']) for m in sg['application']['messages']]}")
