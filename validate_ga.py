"""
GA Validation Script
====================

Comprehensive validation of the Genetic Algorithm implementation.
Tests multiple applications and validates all constraints.

Usage:
    python validate_ga.py
    python validate_ga.py --quick  # Test only small applications
    python validate_ga.py --app Application/T2.json  # Test specific application
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return output"""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr


def run_ga(app_file, repo_root):
    """Run GA on an application"""
    print(f"\n{'='*70}")
    print(f"Running GA on: {app_file}")
    print(f"{'='*70}")
    
    cmd = f"python -m src.main 0 {app_file}"
    returncode, stdout, stderr = run_command(cmd, cwd=repo_root)
    
    if returncode != 0:
        print(f"❌ GA FAILED with return code {returncode}")
        print("STDERR:", stderr[-500:] if len(stderr) > 500 else stderr)
        return False
    
    print("✅ GA completed successfully")
    return True


def create_solution(app_file, repo_root):
    """Create simplified solution from GA logs"""
    cmd = f"python src/simplify.py --input {app_file} --log Logs/global_ga.log"
    returncode, stdout, stderr = run_command(cmd, cwd=repo_root)
    
    if returncode != 0:
        print(f"❌ Solution creation FAILED")
        return False
    
    print("✅ Solution created")
    return True


def validate_solution(solution_file, app_file, repo_root):
    """Validate a solution"""
    print(f"\nValidating: {solution_file}")
    
    cmd = f"python Script/check_solutions.py --solution {solution_file} --application {app_file}"
    returncode, stdout, stderr = run_command(cmd, cwd=repo_root)
    
    print(stdout)
    
    # Check if solution is valid
    if "Valid: YES" in stdout:
        # Extract makespan
        for line in stdout.split('\n'):
            if line.startswith("Makespan:"):
                makespan = line.split(':')[1].strip()
                print(f"✅ VALID - Makespan: {makespan}")
                return True, makespan
    
    print("❌ VALIDATION FAILED")
    return False, None


def main():
    parser = argparse.ArgumentParser(description='Validate GA implementation')
    parser.add_argument('--quick', action='store_true', help='Test only small applications')
    parser.add_argument('--app', type=str, help='Test specific application file')
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent
    
    # Define test applications
    if args.app:
        test_apps = [args.app]
    elif args.quick:
        test_apps = [
            'Application/T2.json',
            'Application/T2_var_001.json',
        ]
    else:
        test_apps = [
            'Application/T2.json',
            'Application/T20.json',
            'Application/T2_var_001.json',
            'Application/T2_var_005.json',
            'Application/example_N5.json',
        ]
    
    print("="*70)
    print("GA VALIDATION TEST SUITE")
    print("="*70)
    print(f"Testing {len(test_apps)} application(s)")
    print()
    
    results = []
    
    for app_file in test_apps:
        app_name = Path(app_file).stem
        solution_file = f"solution/{app_name}_ga.json"
        
        # Run GA
        ga_success = run_ga(app_file, repo_root)
        if not ga_success:
            results.append((app_name, False, "GA Failed", None))
            continue
        
        # Create solution
        sol_success = create_solution(app_file, repo_root)
        if not sol_success:
            results.append((app_name, False, "Solution Creation Failed", None))
            continue
        
        # Validate solution
        valid, makespan = validate_solution(solution_file, app_file, repo_root)
        results.append((app_name, valid, "Valid" if valid else "Invalid", makespan))
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, valid, _, _ in results if valid)
    total = len(results)
    
    for app_name, valid, status, makespan in results:
        symbol = "✅" if valid else "❌"
        makespan_str = f"Makespan: {makespan}" if makespan else ""
        print(f"{symbol} {app_name:30s} {status:20s} {makespan_str}")
    
    print(f"\n{passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED! 🎉")
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
