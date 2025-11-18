"""Generate GA solutions for all applications in batch with progress tracking"""
import subprocess
import json
import glob
import time
from pathlib import Path
from datetime import datetime
import sys

def run_ga_for_application(app_path, seed=0, timeout=300):
    """Run GA for a single application with specific seed"""
    try:
        # Run main GA with seed
        cmd = ['python', '-m', 'src.main', str(seed), app_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            return False, f"GA failed: {result.stderr[-200:]}"
        
        # Call simplify.py to extract solution from log
        simplify_cmd = ['python', 'src/simplify.py', '--input', app_path, '--seed', str(seed)]
        simplify_result = subprocess.run(simplify_cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
        
        if simplify_result.returncode != 0:
            return False, f"Simplify failed: {simplify_result.stderr[-200:]}"
        
        # Check if solution file was created
        app_name = Path(app_path).stem
        if seed == 0:
            solution_path = f'solution/{app_name}_ga.json'
        else:
            solution_path = f'solution/{app_name}_seed{seed:02d}_ga.json'
        
        if not Path(solution_path).exists():
            return False, "Solution file not created"
        
        # Validate solution
        val_cmd = ['python', 'Script/check_solutions.py', '--solution', solution_path, '--application', app_path]
        val_result = subprocess.run(val_cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
        
        # Check if valid (check for "Valid: YES" or "Valid: True")
        if 'Valid: YES' in val_result.stdout or 'valid=True' in val_result.stdout or 'Valid: True' in val_result.stdout:
            return True, "Valid"
        else:
            # Extract validation errors
            lines = val_result.stdout.split('\n')
            errors = [line.strip() for line in lines if 'False' in line or 'violation' in line.lower() or 'FAIL' in line]
            return False, f"Invalid: {' '.join(errors[:2])}"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"

def generate_all_solutions(app_dir='Application', timeout=300, skip_existing=True, num_seeds=5):
    """Generate GA solutions for all applications with multiple seeds
    
    Args:
        app_dir: Directory containing application JSON files
        timeout: Timeout per GA run in seconds
        skip_existing: Skip if solution file already exists
        num_seeds: Number of random seeds to generate per application (default: 5)
    """
    
    # Get all application files
    app_files = sorted(glob.glob(f'{app_dir}/*.json'))
    total_apps = len(app_files)
    total_runs = total_apps * num_seeds
    
    print("="*70)
    print(f"BATCH GA SOLUTION GENERATION (MULTI-SEED)")
    print("="*70)
    print(f"Applications: {total_apps}")
    print(f"Seeds per app: {num_seeds}")
    print(f"Total runs: {total_runs} ({total_apps} x {num_seeds})")
    print(f"Timeout per run: {timeout}s")
    print(f"Skip existing: {skip_existing}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {
        'valid': [],
        'invalid': [],
        'failed': []
    }
    
    start_time = time.time()
    run_count = 0
    
    for app_idx, app_path in enumerate(app_files, 1):
        app_name = Path(app_path).stem
        
        for seed in range(num_seeds):
            run_count += 1
            
            # Determine solution filename
            if seed == 0:
                solution_path = f'solution/{app_name}_ga.json'
                seed_label = f"seed{seed:02d}"
            else:
                solution_path = f'solution/{app_name}_seed{seed:02d}_ga.json'
                seed_label = f"seed{seed:02d}"
            
            # Skip if solution exists and valid
            if skip_existing and Path(solution_path).exists():
                print(f"[{run_count:3d}/{total_runs}] O {app_name:25s} {seed_label} [SKIPPED]")
                continue
            
            print(f"[{run_count:3d}/{total_runs}] > {app_name:25s} {seed_label} ", end='', flush=True)
            
            # Run GA with seed
            success, message = run_ga_for_application(app_path, seed, timeout)
            
            elapsed = time.time() - start_time
            eta = (elapsed / run_count) * (total_runs - run_count)
            
            solution_id = f"{app_name}_seed{seed:02d}"
            
            if success:
                results['valid'].append(solution_id)
                print(f"OK Valid     [ETA: {int(eta//60)}m {int(eta%60)}s]")
            else:
                if "Invalid" in message:
                    results['invalid'].append((solution_id, message))
                    print(f"ERR {message[:35]:35s} [ETA: {int(eta//60)}m {int(eta%60)}s]")
                else:
                    results['failed'].append((solution_id, message))
                    print(f"WARN {message[:35]:35s} [ETA: {int(eta//60)}m {int(eta%60)}s]")
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"Applications: {total_apps}")
    print(f"Seeds per app: {num_seeds}")
    print(f"Total runs: {total_runs}")
    print(f"Valid solutions:    {len(results['valid'])} ({len(results['valid'])/total_runs*100:.1f}%)")
    print(f"Invalid solutions:  {len(results['invalid'])} ({len(results['invalid'])/total_runs*100:.1f}%)")
    print(f"Failed to run:      {len(results['failed'])} ({len(results['failed'])/total_runs*100:.1f}%)")
    print(f"Total time:         {int(total_time//60)}m {int(total_time%60)}s")
    print(f"Avg time/run:       {total_time/total_runs:.1f}s")
    print("="*70)
    
    # Save detailed results
    report = {
        'timestamp': datetime.now().isoformat(),
        'applications': total_apps,
        'seeds_per_app': num_seeds,
        'total_runs': total_runs,
        'valid_count': len(results['valid']),
        'invalid_count': len(results['invalid']),
        'failed_count': len(results['failed']),
        'valid': results['valid'],
        'invalid': [{'solution': app, 'reason': msg} for app, msg in results['invalid']],
        'failed': [{'solution': app, 'reason': msg} for app, msg in results['failed']],
        'total_time_seconds': total_time,
        'avg_time_seconds': total_time / total_runs
    }
    
    report_file = 'ga_generation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Detailed report saved to: {report_file}")
    
    # Show invalid/failed details
    if results['invalid']:
        print(f"\nInvalid solutions ({len(results['invalid'])}):")
        for app, msg in results['invalid'][:10]:
            print(f"  - {app}: {msg[:60]}")
        if len(results['invalid']) > 10:
            print(f"  ... and {len(results['invalid'])-10} more")
    
    if results['failed']:
        print(f"\nFailed executions ({len(results['failed'])}):")
        for app, msg in results['failed'][:10]:
            print(f"  - {app}: {msg[:60]}")
        if len(results['failed']) > 10:
            print(f"  ... and {len(results['failed'])-10} more")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate GA solutions for all applications with multiple seeds')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per GA run (seconds)')
    parser.add_argument('--no-skip', action='store_true', help='Regenerate existing solutions')
    parser.add_argument('--app-dir', type=str, default='Application', help='Application directory')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds per application (default: 5)')
    
    args = parser.parse_args()
    
    results = generate_all_solutions(
        app_dir=args.app_dir,
        timeout=args.timeout,
        skip_existing=not args.no_skip,
        num_seeds=args.seeds
    )
    
    # Exit with error if no valid solutions
    if len(results['valid']) == 0:
        sys.exit(1)
