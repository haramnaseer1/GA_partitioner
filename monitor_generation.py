"""Monitor the GA solution generation progress"""
import os
import time
from pathlib import Path
from datetime import datetime

def monitor_progress():
    """Monitor generation progress"""
    solution_dir = Path('solution')
    
    print("="*70)
    print("GA GENERATION PROGRESS MONITOR")
    print("="*70)
    print(f"Started monitoring: {datetime.now().strftime('%H:%M:%S')}\n")
    
    while True:
        # Count solution files
        if solution_dir.exists():
            solutions = list(solution_dir.glob('*.json'))
            total = len(solutions)
            
            # Expected: 107 apps × 5 seeds = 535
            expected = 535
            percentage = (total / expected) * 100 if expected > 0 else 0
            
            # Clear line and print progress
            print(f"\r✓ Generated: {total}/{expected} ({percentage:.1f}%) | {datetime.now().strftime('%H:%M:%S')}", end='', flush=True)
            
            if total >= expected:
                print(f"\n\n✓ Generation complete! {total} solutions generated.")
                break
        else:
            print("\rWaiting for solution directory...", end='', flush=True)
        
        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
