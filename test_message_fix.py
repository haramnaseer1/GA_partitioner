"""
Quick test to verify the message size bug fix.
Runs one GA iteration and checks if Combined_SubGraphs has correct message sizes.
"""

import json
import os
import sys

# Clean up old files
import shutil
shutil.rmtree('Combined_SubGraphs', ignore_errors=True)
shutil.rmtree('SubGraphs', ignore_errors=True)

print("=" * 70)
print("TESTING MESSAGE SIZE BUG FIX")
print("=" * 70)

# Run GA on one application
print("\n[1/3] Running GA on T2_var_001...")
ret = os.system('python train_model_main.py Application/T2_var_001.json > test_run.log 2>&1')

if ret != 0:
    print("❌ GA run failed!")
    with open('test_run.log') as f:
        print(f.read()[-2000:])  # Last 2000 chars
    sys.exit(1)

print("✓ GA completed")

# Check if Combined_SubGraphs was created
if not os.path.exists('Combined_SubGraphs'):
    print("\n❌ Combined_SubGraphs directory not created!")
    sys.exit(1)

print("\n[2/3] Checking Combined_SubGraphs message sizes...")

# Load original application
with open('Application/T2_var_001.json') as f:
    app = json.load(f)

original_messages = {
    (m['sender'], m['receiver']): m['size'] 
    for m in app['application']['messages']
}

# Check all Combined_SubGraphs files
all_correct = True
total_messages = 0
correct_messages = 0
wrong_messages = []

for filename in os.listdir('Combined_SubGraphs'):
    if not filename.endswith('.json'):
        continue
    
    filepath = os.path.join('Combined_SubGraphs', filename)
    with open(filepath) as f:
        combined = json.load(f)
    
    for msg in combined['application']['messages']:
        total_messages += 1
        sender = msg['sender']
        receiver = msg['receiver']
        size = msg['size']
        
        # Check if this message exists in original
        if (sender, receiver) in original_messages:
            expected_size = original_messages[(sender, receiver)]
            if size == expected_size:
                correct_messages += 1
            else:
                all_correct = False
                wrong_messages.append(
                    f"  File {filename}: Message {sender}->{receiver} has size={size}, "
                    f"expected={expected_size}"
                )
        # Note: Some messages may not exist in original (inter-partition synthetic)

print(f"\n[3/3] Results:")
print(f"  Total messages in Combined_SubGraphs: {total_messages}")
print(f"  Correct message sizes: {correct_messages}")
print(f"  Wrong message sizes: {len(wrong_messages)}")

if wrong_messages:
    print("\n❌ WRONG MESSAGE SIZES FOUND:")
    for msg in wrong_messages[:10]:  # Show first 10
        print(msg)
    if len(wrong_messages) > 10:
        print(f"  ... and {len(wrong_messages) - 10} more")
else:
    print("\n✅ ALL MESSAGE SIZES CORRECT!")

print("\n" + "=" * 70)
if all_correct and total_messages > 0:
    print("✅ BUG FIX VERIFIED - Messages have correct original sizes!")
    sys.exit(0)
else:
    print("❌ BUG STILL PRESENT - Messages still have wrong sizes")
    sys.exit(1)
