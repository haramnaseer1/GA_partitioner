"""
Multi-Seed GA Data Generation Script

Generates multiple solutions per application using different random seeds.
This increases dataset size for better GNN training.

Usage:
    python Script/generate_multi_seed_data.py --seeds 10
    python Script/generate_multi_seed_data.py --seeds 5 --apps Application/T2_var_001.json
"""

import subprocess
import sys
import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

def log_message(log_file, message):
    """Write message to both console and log file"""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_msg = f"{timestamp} {message}"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')
    
    print(log_msg, flush=True)

def update_config(file_name):
    """Update config.py with new application file name"""
    config_path = 'src/config.py'
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith('file_name ='):
            lines[i] = f"file_name = '{file_name}'  # Auto-updated by generate_multi_seed_data.py\n"
            break
    
    with open(config_path, 'w') as f:
        f.writelines(lines)

def run_ga(app_file, seed, log_file):
    """Run GA with specific random seed"""
    log_message(log_file, f"  Running GA with seed={seed}")
    
    # Update config
    update_config(app_file)
    
    # Create unique GA log file for this run to avoid log collisions
    base_name = app_file.replace('.json', '')
    ga_log = f'Logs/ga_{base_name}_seed{seed:02d}.log'
    
    # Clear any existing log for this run
    if os.path.exists(ga_log):
        os.remove(ga_log)
    
    # Set environment variable for random seed
    env = os.environ.copy()
    env['GA_RANDOM_SEED'] = str(seed)
    env['PYTHONHASHSEED'] = str(seed)
    env['GA_LOG_FILE'] = ga_log  # Tell GA to use this log file
    
    # Run main.py using module syntax to handle relative imports
    start_time = time.time()
    result = subprocess.run(
        [sys.executable, '-m', 'src.main', '0'],
        capture_output=True,
        text=True,
        timeout=900,  # 15 minute timeout (GA can take 8-10 minutes)
        env=env
    )
    duration = time.time() - start_time
    
    if result.returncode != 0:
        log_message(log_file, f"    ERROR: GA failed (exit code {result.returncode})")
        log_message(log_file, f"    STDERR: {result.stderr[:500]}")
        return False
    
    log_message(log_file, f"    GA completed in {duration:.1f}s")
    return True, ga_log

def run_simplify(app_file, seed, ga_log, log_file):
    """Run simplify.py to extract solution from the specific GA log"""
    base_name = app_file.replace('.json', '')
    output_name = f"{base_name}_seed{seed:02d}_ga.json"
    
    log_message(log_file, f"  Running simplify to create {output_name}")
    
    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable, 'src/simplify.py',
            '--input', f'Application/{app_file}',
            '--outdir', 'solution',
            '--log', ga_log  # Use the unique GA log for this run
        ],
        capture_output=True,
        text=True,
        timeout=60
    )
    duration = time.time() - start_time
    
    if result.returncode != 0:
        log_message(log_file, f"    ERROR: Simplify failed")
        log_message(log_file, f"    STDOUT: {result.stdout[:300]}")
        log_message(log_file, f"    STDERR: {result.stderr[:300]}")
        return False
    
    # Rename output to include seed
    default_output = f"solution/{base_name}_ga.json"
    seeded_output = f"solution/{output_name}"
    
    if os.path.exists(default_output):
        if os.path.exists(seeded_output):
            os.remove(seeded_output)
        os.rename(default_output, seeded_output)
        log_message(log_file, f"    Created: {seeded_output}")
    else:
        log_message(log_file, f"    ERROR: Output file not found: {default_output}")
        return False
    
    return True

def validate_solution(solution_file, log_file):
    """Validate solution against constraints"""
    result = subprocess.run(
        [sys.executable, 'Script/check_solutions.py', '--solution', solution_file],
        capture_output=True,
        text=True
    )
    
    is_valid = "Valid: YES" in result.stdout or "Eligibility Constraints: YES" in result.stdout
    
    if not is_valid:
        # Count violations
        prec_violations = result.stdout.count("Precedence violation")
        overlap_violations = result.stdout.count("Overlap on processor")
        log_message(log_file, f"    INVALID: {prec_violations} precedence, {overlap_violations} overlap violations")
    else:
        log_message(log_file, f"    VALID: All constraints satisfied!")
    
    return is_valid

def main():
    parser = argparse.ArgumentParser(description='Generate multiple GA solutions per application')
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds per app')
    parser.add_argument('--apps', nargs='+', help='Specific apps to process (default: all)')
    parser.add_argument('--validate', action='store_true', help='Validate solutions after generation')
    args = parser.parse_args()
    
    # Setup
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    app_dir = Path('Application')
    solution_dir = Path('solution')
    log_dir = Path('Logs')
    
    # Create directories
    solution_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = log_dir / f"multi_seed_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Get application files
    if args.apps:
        app_files = [Path(a).name for a in args.apps]
    else:
        app_files = sorted([f.name for f in app_dir.glob('T2*.json')])
    
    log_message(log_file, "="*100)
    log_message(log_file, "MULTI-SEED GA DATA GENERATION")
    log_message(log_file, "="*100)
    log_message(log_file, f"Applications: {len(app_files)}")
    log_message(log_file, f"Seeds per app: {args.seeds}")
    log_message(log_file, f"Total solutions to generate: {len(app_files) * args.seeds}")
    log_message(log_file, f"Validation: {'ENABLED' if args.validate else 'DISABLED'}")
    log_message(log_file, "="*100)
    
    stats = {
        'total': len(app_files) * args.seeds,
        'success': 0,
        'failed': 0,
        'valid': 0,
        'invalid': 0
    }
    
    for app_idx, app_file in enumerate(app_files, 1):
        log_message(log_file, f"\n[{app_idx}/{len(app_files)}] Processing: {app_file}")
        log_message(log_file, "-"*100)
        
        for seed in range(args.seeds):
            log_message(log_file, f"Seed {seed+1}/{args.seeds}")
            
            try:
                # Run GA
                ga_result = run_ga(app_file, seed, log_file)
                if isinstance(ga_result, tuple):
                    ga_success, ga_log = ga_result
                else:
                    ga_success = ga_result
                    ga_log = 'Logs/global_ga.log'  # Fallback to default
                
                if not ga_success:
                    stats['failed'] += 1
                    continue
                
                # Extract solution
                if not run_simplify(app_file, seed, ga_log, log_file):
                    stats['failed'] += 1
                    continue
                
                stats['success'] += 1
                
                # Validate if requested
                if args.validate:
                    base_name = app_file.replace('.json', '')
                    solution_file = f"solution/{base_name}_seed{seed:02d}_ga.json"
                    
                    if validate_solution(solution_file, log_file):
                        stats['valid'] += 1
                    else:
                        stats['invalid'] += 1
            
            except Exception as e:
                log_message(log_file, f"    EXCEPTION: {e}")
                stats['failed'] += 1
    
    # Summary
    log_message(log_file, "\n" + "="*100)
    log_message(log_file, "GENERATION COMPLETE")
    log_message(log_file, "="*100)
    log_message(log_file, f"Total attempted: {stats['total']}")
    log_message(log_file, f"Successful: {stats['success']}")
    log_message(log_file, f"Failed: {stats['failed']}")
    
    if args.validate:
        log_message(log_file, f"Valid solutions: {stats['valid']}")
        log_message(log_file, f"Invalid solutions: {stats['invalid']}")
        log_message(log_file, f"Validation rate: {100*stats['valid']/max(1,stats['success']):.1f}%")
    
    log_message(log_file, f"\nLog saved to: {log_file}")
    
    return 0 if stats['failed'] == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
