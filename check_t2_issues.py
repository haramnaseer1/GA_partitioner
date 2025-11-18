import json

# Check original T2.json messages
with open('Application/T2.json') as f:
    app = json.load(f)

print("Original T2.json application:")
print(f"  Total tasks: {len(app['application']['jobs'])}")
print(f"  Total messages: {len(app['application']['messages'])}")

# Find messages between tasks 0, 1, 2
msgs = [m for m in app['application']['messages'] 
        if m['sender'] in [0,1,2] and m['receiver'] in [0,1,2]]

print(f"\nMessages between tasks 0, 1, 2:")
for m in msgs:
    print(f"  Task {m['sender']}->{m['receiver']}: size={m['size']}")

# Check the solution file issues
print("\n" + "="*70)
print("CHECKING ISSUES FROM YOUR DOCUMENT:")
print("="*70)

# Issue 1: Message size 24 vs 25
msg_0_to_1 = [m for m in app['application']['messages'] if m['sender']==0 and m['receiver']==1]
if msg_0_to_1:
    print(f"\n1. Task 0->1 message:")
    print(f"   Original size in T2.json: {msg_0_to_1[0]['size']} bytes")
    print(f"   Your solution showed: 25 bytes")
    print(f"   Status: {'FIXED - will use ' + str(msg_0_to_1[0]['size']) if msg_0_to_1[0]['size'] != 24 else 'BUG REMAINS'}")

# Issue 2: Message size 24 vs 28
msg_0_to_2 = [m for m in app['application']['messages'] if m['sender']==0 and m['receiver']==2]
if msg_0_to_2:
    print(f"\n2. Task 0->2 message:")
    print(f"   Original size in T2.json: {msg_0_to_2[0]['size']} bytes")
    print(f"   Your solution showed: 28 bytes")
    print(f"   Status: {'FIXED - will use ' + str(msg_0_to_2[0]['size']) if msg_0_to_2[0]['size'] != 24 else 'BUG REMAINS'}")

# Issue 3: Message size Task 2->3
msg_2_to_3 = [m for m in app['application']['messages'] if m['sender']==2 and m['receiver']==3]
if msg_2_to_3:
    print(f"\n3. Task 2->3 message:")
    print(f"   Original size in T2.json: {msg_2_to_3[0]['size']} bytes")
    print(f"   Your solution showed: 0 bytes")
    print(f"   Status: {'FIXED - will use ' + str(msg_2_to_3[0]['size']) if msg_2_to_3[0]['size'] != 0 else 'BUG REMAINS'}")
else:
    print(f"\n3. Task 2->3 message:")
    print(f"   Status: NO MESSAGE EXISTS in original application!")
    print(f"   This might be a scheduling artifact, not a data issue")
