"""
Test the platform fix on multiple applications.
Verify each uses the correct platform and produces valid solutions.
"""
import subprocess
import json
import sys
import os

test_apps = [
    ("T2_var_001.json", 2, list(range(21, 27))),
    ("T2_var_005.json", 2, list(range(21, 27))),
    ("T5_var_001.json", 5, list(range(51, 55))),
]

print("Testing platform fix on multiple applications...")
print("=" * 70)

results = []

for app_name, expected_platform, expected_proc_range in test_apps:
    print(f"\nTesting: {app_name}")
    print("-" * 70)
    
    # Run GA
    print(f"  Running GA...")
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "local", app_name],
        cwd="d:/Hira/Freelance/ARman/GA_Partitioner",
        capture_output=True,
        text=True,
        timeout=180
    )
    
    if result.returncode != 0:
        print(f"  ✗ GA FAILED")
        results.append((app_name, False, "GA failed"))
        continue
    
    # Check if correct platform was used (from stdout)
    if f"Using Platform {expected_platform}" in result.stdout:
        print(f"  ✓ Used Platform {expected_platform}")
    else:
        print(f"  ✗ Did NOT use Platform {expected_platform}")
        results.append((app_name, False, f"Wrong platform (expected {expected_platform})"))
        continue
    
    # Extract solution
    print(f"  Extracting solution...")
    subprocess.run(
        [sys.executable, "src/simplify.py", "--input", f"Application/{app_name}", "--log", "Logs/global_ga.log"],
        cwd="d:/Hira/Freelance/ARman/GA_Partitioner",
        capture_output=True
    )
    
    # Check processor IDs
    sol_path = f"d:/Hira/Freelance/ARman/GA_Partitioner/solution/{app_name.replace('.json', '_ga.json')}"
    with open(sol_path) as f:
        sol = json.load(f)
        proc_ids = sorted(set(t['node_id'] for t in sol))
    
    if set(proc_ids).issubset(set(expected_proc_range)):
        print(f"  ✓ Processors {proc_ids} in expected range {expected_proc_range}")
    else:
        print(f"  ✗ Processors {proc_ids} NOT in expected range {expected_proc_range}")
        results.append((app_name, False, f"Wrong processors: {proc_ids}"))
        continue
    
    # Validate solution
    print(f"  Validating solution...")
    val_result = subprocess.run(
        [sys.executable, "Script/check_solutions.py", "--solution", sol_path.replace("d:/Hira/Freelance/ARman/GA_Partitioner/", "")],
        cwd="d:/Hira/Freelance/ARman/GA_Partitioner",
        capture_output=True,
        text=True
    )
    
    if "Valid: YES" in val_result.stdout:
        print(f"  ✓ Solution VALID")
        results.append((app_name, True, "All checks passed"))
    else:
        print(f"  ✗ Solution INVALID")
        results.append((app_name, False, "Validation failed"))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for app, success, msg in results:
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status:8} {app:20} - {msg}")

total = len(results)
passed = sum(1 for _, s, _ in results if s)
print(f"\nTotal: {passed}/{total} passed ({100*passed//total if total > 0 else 0}%)")
