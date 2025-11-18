"""
Quick test: Run GA on T2_var_003 with fixed platform detection.
Check if solution now uses correct platform processors.
"""
import subprocess
import json
import sys

app_name = "T2_var_003.json"

print(f"Running GA on {app_name} with fixed platform detection...")
print("="*60)

# Run the GA (using main.py entry point)
result = subprocess.run(
    [sys.executable, "-m", "src.main", "local", app_name],
    cwd="d:/Hira/Freelance/ARman/GA_Partitioner",
    capture_output=True,
    text=True,
    timeout=120
)

print("STDOUT:")
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

if result.returncode != 0:
    print("\nSTDERR:")
    print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
    print(f"\n✗ GA failed with return code {result.returncode}")
else:
    print("\n✓ GA completed successfully")
    
    # Check the solution file
    sol_path = f"solution/{app_name.replace('.json', '_ga.json')}"
    try:
        with open(sol_path) as f:
            sol = json.load(f)
            proc_ids = sorted(set(t['node_id'] for t in sol))
            print(f"\nSolution processor IDs: {proc_ids}")
            
            # Check if they match Platform 2 (21-26)
            expected = list(range(21, 27))
            if set(proc_ids).issubset(set(expected)):
                print(f"✓ Processors match Platform 2 expected range {expected}")
            else:
                print(f"✗ Processors DON'T match Platform 2 expected range {expected}")
                print(f"  This might be okay if fewer processors were used")
    except Exception as e:
        print(f"✗ Could not read solution: {e}")
