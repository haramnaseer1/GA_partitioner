"""Quick status check for GNN data preparation pipeline"""
import json
from pathlib import Path
from datetime import datetime

def check_status():
    """Check current status of data preparation"""
    
    print("="*70)
    print("GNN DATA PREPARATION STATUS")
    print("="*70)
    print(f"Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check applications
    app_dir = Path('Application')
    apps = list(app_dir.glob('*.json')) if app_dir.exists() else []
    print(f"📁 Applications: {len(apps)} files")
    
    # Check solutions
    sol_dir = Path('solution')
    solutions = list(sol_dir.glob('*_ga.json')) if sol_dir.exists() else []
    print(f"📁 GA Solutions: {len(solutions)} files")
    
    if apps and solutions:
        coverage = len(solutions) / len(apps) * 100
        print(f"   Coverage: {coverage:.1f}%")
    
    # Check training data
    training_file = Path('training_data.pt')
    if training_file.exists():
        size_mb = training_file.stat().st_size / (1024*1024)
        print(f"📁 Training Data: training_data.pt ({size_mb:.2f} MB)")
    else:
        print(f"📁 Training Data: Not created yet")
    
    # Check generation report
    report_file = Path('ga_generation_report.json')
    if report_file.exists():
        with open(report_file) as f:
            report = json.load(f)
        
        print(f"\n📊 Last Generation Report:")
        print(f"   Timestamp: {report.get('timestamp', 'Unknown')}")
        print(f"   Total apps: {report.get('total', 0)}")
        print(f"   Valid: {report.get('valid_count', 0)} ({report.get('valid_count', 0)/report.get('total', 1)*100:.1f}%)")
        print(f"   Invalid: {report.get('invalid_count', 0)}")
        print(f"   Failed: {report.get('failed_count', 0)}")
        print(f"   Avg time: {report.get('avg_time_seconds', 0):.1f}s per app")
        
        if report.get('invalid'):
            print(f"\n   Invalid apps:")
            for item in report['invalid'][:5]:
                print(f"     • {item['app']}: {item['reason'][:50]}")
            if len(report['invalid']) > 5:
                print(f"     ... and {len(report['invalid'])-5} more")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("="*70)
    
    if len(solutions) == 0:
        print("⚠ No GA solutions found")
        print("  → Run: python generate_all_ga_solutions.py")
    elif len(solutions) < len(apps):
        print(f"⚠ Only {len(solutions)}/{len(apps)} solutions generated")
        print("  → Run: python generate_all_ga_solutions.py --no-skip")
    else:
        print("✓ All applications have GA solutions")
    
    if not training_file.exists():
        print("⚠ Training data not created")
        print("  → Run: python create_tensors.py")
    else:
        print("✓ Training data ready for GNN training")
    
    # Quick start
    print(f"\n{'='*70}")
    print("QUICK START")
    print("="*70)
    print("Generate everything:")
    print("  python prepare_gnn_data.py")
    print("\nOr step by step:")
    print("  1. python generate_all_ga_solutions.py")
    print("  2. python create_tensors.py")
    print("  3. python verify_tensors.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    check_status()
