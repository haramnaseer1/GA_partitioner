import json

# Load solution and application
sol = json.load(open('solution/T2_var_001_ga.json'))
app = json.load(open('Application/T2_var_001.json'))

# Create message lookup
msg_lookup = {}
for msg in app['application']['messages']:
    key = (msg['sender'], msg['receiver'])
    msg_lookup[key] = msg

print("=== Checking Message Size Preservation ===\n")

# Check all dependencies
total_deps = 0
correct = 0
wrong = 0

for task in sol:
    if 'dependencies' in task and task['dependencies']:
        for dep in task['dependencies']:
            total_deps += 1
            sender = dep['task_id']
            receiver = task['task_id']
            dep_msg_size = dep['message_size']
            
            # Find original message
            if (sender, receiver) in msg_lookup:
                orig_msg = msg_lookup[(sender, receiver)]
                orig_size = orig_msg['size']
                
                # Find nodes
                sender_task = [t for t in sol if t['task_id'] == sender][0]
                sender_node = sender_task['node_id']
                receiver_node = task['node_id']
                same_node = sender_node == receiver_node
                
                if dep_msg_size == orig_size:
                    print(f"✓ Task {receiver} <- {sender} (nodes {sender_node}->{receiver_node}): dep_size={dep_msg_size}, orig_size={orig_size} {'[SAME-NODE]' if same_node else '[CROSS-NODE]'}")
                    correct += 1
                else:
                    print(f"✗ Task {receiver} <- {sender} (nodes {sender_node}->{receiver_node}): dep_size={dep_msg_size}, orig_size={orig_size} {'[SAME-NODE]' if same_node else '[CROSS-NODE]'}")
                    wrong += 1
            else:
                print(f"⚠ Task {receiver} <- {sender}: NO MESSAGE FOUND IN APP")

print(f"\n=== Summary ===")
print(f"Total Dependencies: {total_deps}")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
