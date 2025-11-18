import json

# Check what messages SHOULD be in Combined_SubGraphs/1.json based on tasks [0,1,2]
with open('Application/T2_var_001.json') as f:
    app = json.load(f)

tasks_in_partition = [0, 1, 2]
print("Tasks in partition:", tasks_in_partition)
print("\nMessages in original application:")
for msg in app['application']['messages']:
    sender = msg['sender']
    receiver = msg['receiver']
    size = msg['size']
    
    # Check if BOTH sender and receiver are in this partition
    if sender in tasks_in_partition and receiver in tasks_in_partition:
        print(f"  Msg {msg['id']}: {sender}->{receiver}, size={size} [INTRA-PARTITION - should be included]")
    elif sender in tasks_in_partition or receiver in tasks_in_partition:
        print(f"  Msg {msg['id']}: {sender}->{receiver}, size={size} [INTER-PARTITION - will be missing!]")

# Now check what's actually in Combined_SubGraphs
print("\n" + "="*70)
print("What's ACTUALLY in Combined_SubGraphs/1.json:")
with open('Combined_SubGraphs/1.json') as f:
    combined = json.load(f)
    
print(f"Tasks: {[j['id'] for j in combined['application']['jobs']]}")
print(f"Messages:")
for msg in combined['application']['messages']:
    print(f"  Msg {msg.get('id', 'N/A')}: {msg['sender']}->{msg['receiver']}, size={msg['size']}")
