"""Complete pipeline: Generate GA solutions + Convert to tensors for GNN training"""
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def run_command(description, cmd, timeout=None):
    """Run a command and track execution"""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    start = time.time()
    try:
        result = subprocess.run(cmd, timeout=timeout, encoding='utf-8', errors='replace')
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"\n✓ SUCCESS in {int(elapsed//60)}m {int(elapsed%60)}s")
            return True
        else:
            print(f"\n✗ FAILED (exit code {result.returncode}) after {int(elapsed//60)}m {int(elapsed%60)}s")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⚠ TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False

def main():
    """Run complete GNN data preparation pipeline"""
    
    print("="*70)
    print("GNN TRAINING DATA PREPARATION PIPELINE")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fixed GA Version: Platform detection + Overlap resolution + Phantom dep fix")
    print("="*70)
    
    pipeline_start = time.time()
    
    # Step 1: Clean old solutions
    print("\nStep 1: Cleaning old solution files...")
    solution_dir = Path('solution')
    if solution_dir.exists():
        old_files = list(solution_dir.glob('*.json'))
        print(f"Found {len(old_files)} old solution files")
        
        response = input("Delete old solutions? (y/n): ").strip().lower()
        if response == 'y':
            for f in old_files:
                f.unlink()
            print(f"✓ Deleted {len(old_files)} files")
        else:
            print("⊙ Keeping existing solutions (they will be skipped)")
    else:
        solution_dir.mkdir(exist_ok=True)
        print("✓ Created solution directory")
    
    # Step 2: Generate all GA solutions with multiple seeds
    num_seeds = input("\nHow many seeds per application? (default: 5): ").strip()
    num_seeds = int(num_seeds) if num_seeds.isdigit() else 5
    
    print(f"\nGenerating {num_seeds} solutions per application...")
    print(f"Total expected: 107 apps × {num_seeds} seeds = {107 * num_seeds} solutions\n")
    
    success = run_command(
        f"Generate GA solutions for all applications ({num_seeds} seeds each)",
        ['python', 'generate_all_ga_solutions.py', '--timeout', '300', '--seeds', str(num_seeds)],
        timeout=None  # No overall timeout, each app has individual timeout
    )
    
    if not success:
        print("\n✗ GA generation failed. Stopping pipeline.")
        sys.exit(1)
    
    # Check how many solutions were generated
    solutions = list(Path('solution').glob('*_ga.json'))
    print(f"\n✓ Generated {len(solutions)} solution files")
    
    if len(solutions) == 0:
        print("\n✗ No solutions generated. Stopping pipeline.")
        sys.exit(1)
    
    # Step 3: Convert solutions to tensors
    success = run_command(
        "Convert GA solutions to PyTorch tensors",
        ['python', 'create_tensors.py'],
        timeout=600
    )
    
    if not success:
        print("\n⚠ Tensor conversion had issues, but continuing...")
    
    # Check if training data was created
    training_file = Path('training_data.pt')
    if training_file.exists():
        file_size = training_file.stat().st_size / (1024*1024)  # MB
        print(f"\n✓ Training data created: {training_file} ({file_size:.2f} MB)")
    else:
        print(f"\n✗ Training data file not found: {training_file}")
    
    # Step 4: Verify tensor data
    print(f"\n{'='*70}")
    print("STEP: Verify tensor data")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            ['python', 'verify_tensors.py'],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace'
        )
        print(result.stdout)
        if result.returncode == 0:
            print("\n✓ Tensor verification passed")
        else:
            print("\n⚠ Tensor verification had warnings")
    except Exception as e:
        print(f"\n⚠ Could not verify tensors: {e}")
    
    # Pipeline summary
    total_time = time.time() - pipeline_start
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {int(total_time//60)}m {int(total_time%60)}s")
    print(f"\nGenerated files:")
    print(f"  • GA solutions: solution/*_ga.json ({len(solutions)} files)")
    if training_file.exists():
        print(f"  • Training data: {training_file} ({file_size:.2f} MB)")
    print(f"  • Generation report: ga_generation_report.json")
    print(f"\nNext steps:")
    print(f"  1. Review ga_generation_report.json for any issues")
    print(f"  2. Load training_data.pt in your GNN training script")
    print(f"  3. Train your GNN model!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
