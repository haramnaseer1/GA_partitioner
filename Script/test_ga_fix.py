"""
Test script to regenerate a single solution with the fixed GA and validate it.
This bypasses the relative import issues by testing the schedule generation directly.
"""
import subprocess
import sys
import json
import os

# Test application
test_app = "T2_var_001.json"

print("="*80)
print("TESTING GA FIX")
print("="*80)
print(f"\nStep 1: Regenerating solution for {test_app}")
print("-"*80)

# Run GA through the batch generation script
result = subprocess.run(
    [sys.executable, "Script/generate_all_gnn_data.py"],
    capture_output=True,
    text=True,
    timeout=600  # 10 minute timeout
)

if result.returncode != 0:
    print("❌ GA execution failed!")
    print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
    print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
    sys.exit(1)

print("✓ GA completed successfully")

# Check if solution was created
solution_file = f"solution/{test_app.replace('.json', '_ga.json')}"
if not os.path.exists(solution_file):
    print(f"❌ Solution file not found: {solution_file}")
    sys.exit(1)

print(f"✓ Solution file created: {solution_file}")

print(f"\nStep 2: Validating solution")
print("-"*80)

# Run validation
result = subprocess.run(
    [sys.executable, "Script/check_solutions.py", "--solution", solution_file],
    capture_output=True,
    text=True
)

print(result.stdout)

# Parse validation results
if "Valid: ✅ YES" in result.stdout:
    print("\n" + "="*80)
    print("🎉 SUCCESS! GA fix works - solution passes all constraints!")
    print("="*80)
    sys.exit(0)
elif "Valid: ❌ NO" in result.stdout:
    print("\n" + "="*80)
    print("⚠️  Solution still has violations - more fixes needed")
    print("="*80)
    
    # Count violations
    lines = result.stdout.split('\n')
    for line in lines:
        if 'Precedence Constraints:' in line or 'Non-Overlap Constraints:' in line or 'Eligibility Constraints:' in line:
            print(line)
    sys.exit(1)
else:
    print("\n❌ Validation failed to run properly")
    print(result.stderr)
    sys.exit(1)
