"""
Comprehensive GNN Data Generation Script
=========================================

This script processes ALL application files in the Application/ directory and generates
complete GA scheduling solutions for GNN training. It handles:
- Automatic config.py updates for each application
- GA partitioning execution (main.py)
- Solution simplification (simplify.py)
- Skip logic for existing solutions
- Comprehensive logging and error handling
- Progress tracking

Usage:
    python Script/generate_all_gnn_data.py

Output:
    - Solution files in solution/*_ga.json
    - Log file in Logs/gnn_data_generation.log
"""

import os
import sys
import subprocess
from datetime import datetime
import json
from torch_geometric.data import Data

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

def update_config(app_file, config_path):
    """Update the file_name in config.py"""
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Update the file_name line
    for i, line in enumerate(lines):
        if line.strip().startswith('file_name ='):
            lines[i] = f"file_name = '{app_file}'  # Change the file name to the desired application model name\n"
            break
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def log_message(log_file, message):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def check_solution_exists(app_name, solution_dir):
    """Check if solution file already exists"""
    base_name = app_name.replace('.json', '')
    solution_file = os.path.join(solution_dir, f'{base_name}_ga.json')
    return os.path.exists(solution_file)

def get_app_info(app_path):
    """Get basic info about application (jobs, messages count)"""
    try:
        with open(app_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        num_jobs = len(data.get('application', {}).get('jobs', []))
        num_messages = len(data.get('application', {}).get('messages', []))
        return num_jobs, num_messages
    except:
        return None, None

def main():
    """Main execution function"""
    # Setup paths
    ga_partitioner_root = parent_dir
    application_dir = os.path.join(ga_partitioner_root, 'Application')
    solution_dir = os.path.join(ga_partitioner_root, 'solution')
    log_dir = os.path.join(ga_partitioner_root, 'Logs')
    config_path = os.path.join(ga_partitioner_root, 'src', 'config.py')
    
    # Create directories
    os.makedirs(solution_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(log_dir, 'gnn_data_generation.log')
    
    # Clear previous log and write header
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"GNN Data Generation - Complete Dataset Processing\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
    
    # Get all application files
    app_files = sorted([f for f in os.listdir(application_dir) if f.endswith('.json')])
    
    log_message(log_file, f"Found {len(app_files)} application files to process")
    log_message(log_file, f"Application directory: {application_dir}")
    log_message(log_file, f"Solution directory: {solution_dir}")
    log_message(log_file, f"Config file: {config_path}\n")
    
    # Statistics tracking
    stats = {
        'total': len(app_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'success': 0
    }
    
    failed_apps = []
    
    # Process each application
    for idx, app_file in enumerate(app_files, 1):
        log_message(log_file, "\n" + "="*100)
        log_message(log_file, f"[{idx}/{len(app_files)}] Processing: {app_file}")
        log_message(log_file, "="*100)
        
        # Get application info
        app_path = os.path.join(application_dir, app_file)
        num_jobs, num_messages = get_app_info(app_path)
        if num_jobs is not None:
            log_message(log_file, f"Application details: {num_jobs} jobs, {num_messages} messages")
        
        # Check if solution already exists
        if check_solution_exists(app_file, solution_dir):
            log_message(log_file, f"SKIPPED - Solution already exists for {app_file}")
            stats['skipped'] += 1
            continue
        
        try:
            # Step 1: Update config.py
            log_message(log_file, f"Updating config.py with file_name = '{app_file}'")
            update_config(app_file, config_path)
            
            # Step 2: Run GA partitioning (main.py)
            log_message(log_file, f"🚀 Running GA partitioning...")
            main_start = datetime.now()
            
            main_process = subprocess.run(
                [sys.executable, '-m', 'src.main', '0'],
                capture_output=True,
                text=True,
                cwd=ga_partitioner_root,
                timeout=600  # 10 minute timeout per application
            )
            
            main_duration = (datetime.now() - main_start).total_seconds()
            
            if main_process.returncode != 0:
                log_message(log_file, f"❌ ERROR: GA partitioning failed (exit code {main_process.returncode})")
                log_message(log_file, f"Error output: {main_process.stderr[:500]}")  # First 500 chars
                stats['failed'] += 1
                failed_apps.append({
                    'file': app_file,
                    'stage': 'GA partitioning',
                    'error': main_process.stderr[:200]
                })
                continue
            
            log_message(log_file, f"✅ GA partitioning completed in {main_duration:.1f} seconds")
            
            # Step 3: Run simplification (simplify.py)
            log_message(log_file, f"🔄 Running solution simplification...")
            simplify_start = datetime.now()
            
            # Simplify expects path like "Application/filename.json"
            simplify_input_path = f"Application/{app_file}"
            
            simplify_process = subprocess.run(
                [sys.executable, '-m', 'src.simplify', '--input', simplify_input_path],
                capture_output=True,
                text=True,
                cwd=ga_partitioner_root,
                timeout=60  # 1 minute timeout for simplification
            )
            
            simplify_duration = (datetime.now() - simplify_start).total_seconds()
            
            if simplify_process.returncode != 0:
                log_message(log_file, f"❌ ERROR: Simplification failed (exit code {simplify_process.returncode})")
                log_message(log_file, f"Error output: {simplify_process.stderr[:500]}")
                stats['failed'] += 1
                failed_apps.append({
                    'file': app_file,
                    'stage': 'Simplification',
                    'error': simplify_process.stderr[:200]
                })
                continue
            
            log_message(log_file, f"✅ Simplification completed in {simplify_duration:.1f} seconds")
            
            # Verify solution was created
            if check_solution_exists(app_file, solution_dir):
                total_time = main_duration + simplify_duration
                log_message(log_file, f"✅ SUCCESS - {app_file} processed successfully in {total_time:.1f} seconds!")
                stats['success'] += 1
                stats['processed'] += 1
            else:
                log_message(log_file, f"⚠️  WARNING - Process completed but solution file not found")
                stats['failed'] += 1
                failed_apps.append({
                    'file': app_file,
                    'stage': 'Verification',
                    'error': 'Solution file not created despite successful execution'
                })
        
        except subprocess.TimeoutExpired as e:
            log_message(log_file, f"⏱️  TIMEOUT - {app_file} exceeded time limit")
            stats['failed'] += 1
            failed_apps.append({
                'file': app_file,
                'stage': 'Timeout',
                'error': f'Process exceeded {e.timeout} second timeout'
            })
        
        except Exception as e:
            log_message(log_file, f"❌ EXCEPTION - {app_file}: {str(e)}")
            stats['failed'] += 1
            failed_apps.append({
                'file': app_file,
                'stage': 'Exception',
                'error': str(e)[:200]
            })
    
    # Final Summary
    log_message(log_file, "\n\n" + "="*100)
    log_message(log_file, "PROCESSING COMPLETE - FINAL SUMMARY")
    log_message(log_file, "="*100)
    log_message(log_file, f"Total applications: {stats['total']}")
    log_message(log_file, f"✅ Successfully processed: {stats['success']}")
    log_message(log_file, f"⏭️  Skipped (already exist): {stats['skipped']}")
    log_message(log_file, f"❌ Failed: {stats['failed']}")
    log_message(log_file, f"📊 Success rate: {(stats['success'] + stats['skipped']) / stats['total'] * 100:.1f}%")
    
    if failed_apps:
        log_message(log_file, "\n" + "-"*100)
        log_message(log_file, "FAILED APPLICATIONS DETAILS:")
        log_message(log_file, "-"*100)
        for fail in failed_apps:
            log_message(log_file, f"\n❌ {fail['file']}")
            log_message(log_file, f"   Stage: {fail['stage']}")
            log_message(log_file, f"   Error: {fail['error']}")
    
    # Count actual solution files
    actual_solutions = len([f for f in os.listdir(solution_dir) if f.endswith('_ga.json')])
    log_message(log_file, f"\n📁 Total solution files in directory: {actual_solutions}/{stats['total']}")
    
    log_message(log_file, f"\n✅ Complete! Log saved to: {log_file}")
    log_message(log_file, f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(log_file, "="*100 + "\n")
    
    # Return exit code based on results
    if stats['failed'] > 0:
        print(f"\n⚠️  Warning: {stats['failed']} applications failed. Check log for details.")
        return 1
    else:
        print(f"\n🎉 Success! All {stats['total']} applications processed successfully!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

# Future: Convert to PyTorch Geometric format
# from torch_geometric.data import Data

# # Create graph tensors
# for sample in dataset['samples']:
#     # Node features: [num_tasks, feature_dim]
#     # Edge indices: [2, num_messages]
#     # Labels: task-to-node assignments
#     graph = Data(x=node_features, edge_index=edges, y=labels)
