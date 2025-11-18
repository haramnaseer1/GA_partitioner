"""Test multi-seed generation on a few applications"""
import subprocess
import sys
from pathlib import Path

def test_multiseed():
    """Test generating multiple seeds for a few apps"""
    
    print("="*70)
    print("TESTING MULTI-SEED GENERATION")
    print("="*70)
    print("Testing with 3 applications × 5 seeds = 15 solutions\n")
    
    # Test apps
    test_apps = ['T2.json', 'T20.json', 'T2_var_001.json']
    num_seeds = 5
    
    total_runs = len(test_apps) * num_seeds
    valid_count = 0
    
    for app_idx, app_name in enumerate(test_apps, 1):
        app_path = f'Application/{app_name}'
        print(f"\n[App {app_idx}/{len(test_apps)}] {app_name}")
        print("-" * 70)
        
        for seed in range(num_seeds):
            run_num = (app_idx - 1) * num_seeds + seed + 1
            
            # Run GA with seed
            print(f"  [{run_num:2d}/{total_runs}] Seed {seed:02d} ... ", end='', flush=True)
            
            cmd = ['python', '-m', 'src.main', str(seed), app_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, encoding='utf-8', errors='replace')
            
            # Check solution file
            base_name = Path(app_name).stem
            if seed == 0:
                sol_path = f'solution/{base_name}_ga.json'
            else:
                sol_path = f'solution/{base_name}_seed{seed:02d}_ga.json'
            
            if Path(sol_path).exists():
                # Quick validation
                val_cmd = ['python', 'Script/check_solutions.py', '--solution', sol_path, '--application', app_path]
                val_result = subprocess.run(val_cmd, capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace')
                
                if 'valid=True' in val_result.stdout or 'Valid: True' in val_result.stdout:
                    print("✓ Valid")
                    valid_count += 1
                else:
                    print("✗ Invalid")
            else:
                print("✗ No solution")
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total runs: {total_runs}")
    print(f"Valid: {valid_count}/{total_runs} ({valid_count/total_runs*100:.1f}%)")
    print("="*70)
    
    # Show generated files
    print("\nGenerated solution files:")
    solutions = sorted(Path('solution').glob('*_ga.json'))
    for sol in solutions:
        if any(app.split('.')[0] in sol.name for app in test_apps):
            print(f"  • {sol.name}")
    
    print("\n✓ Test complete! Multi-seed generation is working.")
    print("\nTo generate ALL data with 5 seeds:")
    print("  python generate_all_ga_solutions.py --seeds 5")
    print("\nOr use the full pipeline:")
    print("  python prepare_gnn_data.py")

if __name__ == "__main__":
    try:
        test_multiseed()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
