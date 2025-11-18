"""Generate GA solutions for all applications in batch with progress tracking"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
import time
import json
import glob
from datetime import datetime

def run_ga_for_application(app_path, seed=0, timeout=300):
    """Run GA for a single application with a specific seed"""
    try:
        # Run main GA with seed
        cmd = [sys.executable, '-m', 'src.main', str(seed), app_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            return False, f"GA failed: {result.stderr}"
        
        # Call simplify.py to extract solution from log
        simplify_cmd = [sys.executable, 'src/simplify.py', '--input', app_path, '--seed', str(seed)]
        simplify_result = subprocess.run(simplify_cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
        
        if simplify_result.returncode != 0:
            return False, f"Simplify failed: {simplify_result.stderr[-200:]}"
        
        # Determine the correct solution path based on the seed
        app_name = Path(app_path).stem
        if seed == 0:
            solution_path = f'solution/{app_name}_ga.json'
        else:
            solution_path = f'solution/{app_name}_seed{seed:02d}_ga.json'
        
        if not Path(solution_path).exists():
            return False, "Solution file not created"
        
        # Validate solution
        val_cmd = [sys.executable, 'Script/check_solutions.py', '--solution', solution_path, '--application', app_path]
        val_result = subprocess.run(val_cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
        
        # Check for validity
        if 'Valid: YES' in val_result.stdout or 'valid=True' in val_result.stdout or 'Valid: True' in val_result.stdout:
            return True, "Valid"
        else:
            # Return the full output for detailed debugging
            error_details = f"STDOUT:\n{val_result.stdout}\nSTDERR:\n{val_result.stderr}"
            return False, f"Invalid:\n{error_details}"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"

def generate_all_solutions(app_dir='Application', timeout=300, skip_existing=True, num_seeds=1, limit=None):
    """
    Generate GA solutions for all applications with multiple seeds.
    A num_seeds value of 1 will run with seed 0.
    """
    
    app_files = sorted(glob.glob(f'{app_dir}/T*.json')) # Focus on T-series files
    if limit:
        print(f"--- LIMITING TO {limit} APPLICATIONS ---")
        app_files = app_files[:limit]
    total_apps = len(app_files)
    
    # If num_seeds is 1, we run with seed 0. Otherwise, seeds from 1 to num_seeds.
    seeds_to_run = [0] if num_seeds == 1 else range(1, num_seeds + 1)
    total_runs = total_apps * len(seeds_to_run)
    
    print("="*70)
    print(f"BATCH GA SOLUTION GENERATION")
    print("="*70)
    print(f"Applications to process: {total_apps}")
    print(f"Seeds to run: {seeds_to_run}")
    print(f"Total runs: {total_runs}")
    print(f"Timeout per run: {timeout}s")
    print(f"Skip existing: {skip_existing}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {
        'valid': [],
        'invalid': [],
        'failed': [],
        'skipped': []
    }
    
    start_time = time.time()
    run_count = 0
    
    for app_idx, app_path in enumerate(app_files, 1):
        app_name = Path(app_path).stem
        
        for seed in seeds_to_run:
            run_count += 1
            
            # Determine solution filename and seed label
            if seed == 0:
                solution_path = f'solution/{app_name}_ga.json'
                seed_label = "seed0"
            else:
                solution_path = f'solution/{app_name}_seed{seed:02d}_ga.json'
                seed_label = f"seed{seed:02d}"
            
            # Skip if solution exists
            if skip_existing and Path(solution_path).exists():
                print(f"[{run_count:3d}/{total_runs}] O {app_name:25s} {seed_label} [SKIPPED]")
                results['skipped'].append(f"{app_name}_{seed_label}")
                continue
            
            print(f"[{run_count:3d}/{total_runs}] > {app_name:25s} {seed_label} ", end='', flush=True)
            
            # Run GA
            success, message = run_ga_for_application(app_path, seed, timeout)
            
            elapsed = time.time() - start_time
            eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 0 else 0
            
            solution_id = f"{app_name}_{seed_label}"
            
            if success:
                results['success'].append(solution_id)
                print(f"OK Valid     [ETA: {int(eta//60)}m {int(eta%60)}s]")
            else:
                if "Invalid" in message:
                    results['invalid'].append((solution_id, message))
                else:
                    results['failed'].append((solution_id, message))
                # Print the full error message without truncation for debugging
                # Encode to ASCII with error replacement to avoid UnicodeEncodeError
                safe_message = message.encode('ascii', errors='replace').decode('ascii')
                print(f"ERR {safe_message} [ETA: {int(eta//60)}m {int(eta%60)}s]")
        
        # Break if limit is reached
        if limit is not None and app_idx >= limit:
            print(f"Limit of {limit} applications reached.")
            break
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"Total runs planned: {total_runs}")
    print(f"Valid solutions:    {len(results['valid'])}")
    print(f"Skipped solutions:  {len(results['skipped'])}")
    print(f"Invalid solutions:  {len(results['invalid'])}")
    print(f"Failed runs:        {len(results['failed'])}")
    print(f"Total time:         {int(total_time//60)}m {int(total_time%60)}s")
    if run_count > len(results['skipped']):
        avg_time = total_time / (run_count - len(results['skipped']))
        print(f"Avg time/run:       {avg_time:.1f}s")
    print("="*70)
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {'apps': total_apps, 'seeds': num_seeds, 'timeout': timeout},
        'summary': {
            'valid': len(results['valid']),
            'invalid': len(results['invalid']),
            'failed': len(results['failed']),
            'skipped': len(results['skipped'])
        },
        'invalid_details': results['invalid'],
        'failed_details': results['failed'],
    }
    report_path = f"solution/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Detailed report saved to {report_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch generate GA solutions for all applications.")
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds per application. If 1, runs with seed 0.')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per GA run in seconds.')
    parser.add_argument('--no-skip', action='store_true', help='Force regeneration even if solution exists.')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of applications to process.')
    
    args = parser.parse_args()
    
    generate_all_solutions(
        timeout=args.timeout,
        skip_existing=not args.no_skip,
        num_seeds=args.seeds,
        limit=args.limit
    )

if __name__ == '__main__':
    sys.exit(main())
