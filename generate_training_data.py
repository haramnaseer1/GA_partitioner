"""
Complete Training Data Generation Pipeline
===========================================

Generates GA solutions and converts them to multi-task training tensors.

Usage:
    python generate_training_data.py                    # Default: 5 seeds, 50/30 iterations
    python generate_training_data.py --seeds 3          # Custom seeds
    python generate_training_data.py --quick            # Quick test: 1 seed, 2/2 iterations
    python generate_training_data.py --regenerate       # Skip GA, regenerate tensors only
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
import shutil


def check_prerequisites():
    """Check if required directories and files exist"""
    required_dirs = ['Application', 'Platform', 'src']
    required_files = ['src/global_GA.py', 'src/auxiliary_fun_GA.py', 'create_tensors_multitask.py']
    
    missing = []
    for d in required_dirs:
        if not os.path.exists(d):
            missing.append(f"Directory: {d}")
    
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f"File: {f}")
    
    if missing:
        print("ERROR: Missing prerequisites:")
        for item in missing:
            print(f"  - {item}")
        return False
    
    return True


def count_applications():
    """Count available application files"""
    app_dir = Path('Application')
    json_files = list(app_dir.glob('T*.json'))
    return len(json_files)


def update_config(iterations_gca, iterations_lga):
    """Update config.py with iteration counts"""
    config_path = 'src/config.py'
    
    if not os.path.exists(config_path):
        print(f"WARNING: {config_path} not found, skipping config update")
        return
    
    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Update iteration values
    updated = False
    for i, line in enumerate(lines):
        if 'NUMBER_OF_GENERATIONS_GCA' in line and '=' in line:
            lines[i] = f'NUMBER_OF_GENERATIONS_GCA = {iterations_gca}\n'
            updated = True
            print(f"  Updated GCA iterations: {iterations_gca}")
        elif 'NUMBER_OF_GENERATIONS_LGA' in line and '=' in line:
            lines[i] = f'NUMBER_OF_GENERATIONS_LGA = {iterations_lga}\n'
            updated = True
            print(f"  Updated LGA iterations: {iterations_lga}")
    
    if updated:
        with open(config_path, 'w') as f:
            f.writelines(lines)
        print("  Config updated successfully")
    else:
        print("  WARNING: Could not find iteration settings in config.py")


def run_ga_generation(num_seeds, timeout, no_skip, limit=None):
    """Run GA solution generation"""
    print("\n" + "="*70)
    print("STEP 1: GENERATING GA SOLUTIONS")
    print("="*70)
    
    cmd = [
        sys.executable,
        'generate_all_ga_solutions.py',
        '--seeds', str(num_seeds),
        '--timeout', str(timeout)
    ]
    
    if no_skip:
        cmd.append('--no-skip')

    if limit:
        cmd.extend(['--limit', str(limit)])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"This will run GA for all applications with {num_seeds} seeds")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n[OK] GA solution generation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] GA generation failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n[STOPPED] GA generation interrupted by user")
        return False


def run_tensor_generation(output_path='training_data_multitask.pt'):
    """Run multi-task tensor generation"""
    print("\n" + "="*70)
    print("STEP 2: GENERATING MULTI-TASK TRAINING TENSORS")
    print("="*70)
    
    cmd = [
        sys.executable, 
        'create_tensors_multitask.py',
        '--output', output_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Converting GA solutions to PyTorch Geometric format...")
    print(f"Output: {output_path}")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n[OK] Tensor generation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Tensor generation failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Tensor generation interrupted by user")
        return False


def verify_output(output_path='training_data_multitask.pt'):
    """Verify generated files"""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Check solution directory
    solution_dir = Path('solution')
    if solution_dir.exists():
        ga_solutions = list(solution_dir.glob('*_ga.json'))
        print(f"[OK] GA solutions: {len(ga_solutions)} files in solution/")
    else:
        print("[ERROR] No solution/ directory found")
        ga_solutions = []
    
    # Check training data
    training_file = Path(output_path)
    if training_file.exists():
        import torch
        data = torch.load(training_file, weights_only=False)
        print(f"[OK] Training data: {len(data)} graphs in {output_path}")
        
        if len(data) > 0:
            sample = data[0]
            print(f"  Sample graph: {sample.x.shape[0]} nodes, {sample.edge_index.shape[1]} edges")
    else:
        print(f"[ERROR] No {output_path} found")
    
    print("="*70)
    
    return len(ga_solutions) > 0 and training_file.exists()


def main():
    parser = argparse.ArgumentParser(
        description='Generate complete training dataset for multi-task GNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_training_data.py                    # Full generation (5 seeds, 50/30 iter)
  python generate_training_data.py --seeds 3          # 3 seeds instead of 5
  python generate_training_data.py --quick            # Quick test (1 seed, 2/2 iter)
  python generate_training_data.py --regenerate       # Regenerate tensors only
  python generate_training_data.py --gca 30 --lga 20  # Custom iterations
        """
    )
    
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds per application (default: 5)')
    parser.add_argument('--gca', type=int, default=50,
                        help='GCA iterations (default: 50)')
    parser.add_argument('--lga', type=int, default=30,
                        help='LGA iterations (default: 30)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per GA run in seconds (default: 300)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode: 1 seed, 2/2 iterations')
    parser.add_argument('--regenerate', action='store_true',
                        help='Skip GA generation, only regenerate tensors from existing solutions')
    parser.add_argument('--no-skip', action='store_true',
                        help='Do not skip existing solutions, regenerate all')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of applications to process for debugging')
    parser.add_argument('--output', '-o', type=str, default='training_data_multitask.pt',
                        help='Output path for training tensors (default: training_data_multitask.pt)')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.seeds = 1
        args.gca = 2
        args.lga = 2
        print("\n*** QUICK TEST MODE ***")
        print(f"Seeds: {args.seeds}, GCA: {args.gca}, LGA: {args.lga}\n")
    
    # Header
    print("\n" + "="*70)
    print("MULTI-TASK GNN TRAINING DATA GENERATION")
    print("="*70)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nPlease ensure all required files are present.")
        return 1
    
    # Count applications
    num_apps = count_applications()
    print(f"\nFound {num_apps} application files")
    
    if num_apps == 0:
        print("ERROR: No application files found in Application/ directory")
        return 1
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Seeds per application: {args.seeds}")
    print(f"  GCA iterations: {args.gca}")
    print(f"  LGA iterations: {args.lga}")
    print(f"  Timeout per run: {args.timeout}s")
    print(f"  Total GA runs: {num_apps * args.seeds}")
    print(f"  Skip existing: {not args.no_skip}")
    print(f"  Output file: {args.output}")
    
    # Update config file
    if not args.regenerate:
        print("\nUpdating src/config.py...")
        update_config(args.gca, args.lga)
    
    # Pipeline execution
    success = True
    
    if not args.regenerate:
        # Step 1: Generate GA solutions
        success = run_ga_generation(args.seeds, args.timeout, args.no_skip, args.limit)
    else:
        print("\n>>> Skipping GA generation (--regenerate mode)")
        success = True
    
    if success:
        # Step 2: Generate tensors
        success = run_tensor_generation(output_path=args.output)
    
    if success:
        # Verify output
        success = verify_output(output_path=args.output)
    
    # Summary
    print("\n" + "="*70)
    if success:
        print("[SUCCESS] DATA GENERATION COMPLETED")
        print("="*70)
        print("\nNext step: Train the model")
        print("  python train_model.py --epochs 100")
        return 0
    else:
        print("[FAILED] DATA GENERATION FAILED")
        print("="*70)
        print("\nPlease check the error messages above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
