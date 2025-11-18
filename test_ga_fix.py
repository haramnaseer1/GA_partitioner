"""
Simple test: Run GA directly using the fixed code
"""
import json
import os
import shutil

# Clean up
shutil.rmtree('Combined_SubGraphs', ignore_errors=True)
shutil.rmtree('SubGraphs', ignore_errors=True)
shutil.rmtree('solution', ignore_errors=True)
os.makedirs('solution', exist_ok=True)

print("Running GA on T2_var_001.json...")
print("=" * 70)

# Run via generate_all_ga_solutions with just one file
ret = os.system('python generate_all_ga_solutions.py --apps Application/T2_var_001.json --seeds 1 --timeout 120 --no-skip')

if ret != 0:
    print("\n❌ GA failed")
    exit(1)

print("\n" + "=" * 70)
print("Checking solution file...")

solution_file = 'solution/T2_var_001_seed00_ga.json'
if not os.path.exists(solution_file):
    print(f"❌ Solution file not found: {solution_file}")
    exit(1)

# Load solution
with open(solution_file) as f:
    solution = json.load(f)

# Load original application
with open('Application/T2_var_001.json') as f:
    app = json.load(f)

original_messages = {
    (m['sender'], m['receiver']): m['size'] 
    for m in app['application']['messages']
}

# Check solution dependencies for correct message sizes
print("\nChecking message sizes in solution dependencies...")
total_deps = 0
correct_sizes = 0
wrong_sizes = []

if isinstance(solution, list):
    tasks = solution
else:
    tasks = solution.get('solution', [])

for task in tasks:
    for dep in task.get('dependencies', []):
        total_deps += 1
        sender = dep['task_id']
        receiver = task['task_id']
        msg_size = dep.get('message_size', 0)
        
        if (sender, receiver) in original_messages:
            expected = original_messages[(sender, receiver)]
            if abs(msg_size - expected) < 0.01:  # Allow floating point error
                correct_sizes += 1
            else:
                wrong_sizes.append(
                    f"  Task {sender}->{receiver}: size={msg_size:.2f}, expected={expected}"
                )

print(f"\nResults:")
print(f"  Total dependencies: {total_deps}")
print(f"  Correct message sizes: {correct_sizes}")
print(f"  Wrong message sizes: {len(wrong_sizes)}")

if wrong_sizes:
    print("\n❌ WRONG SIZES:")
    for msg in wrong_sizes[:10]:
        print(msg)
else:
    print("\n✅ ALL MESSAGE SIZES CORRECT!")

if correct_sizes == total_deps and total_deps > 0:
    print("\n✅ BUG FIX VERIFIED!")
    exit(0)
else:
    print("\n❌ BUG STILL PRESENT")
    exit(1)
