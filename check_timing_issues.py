"""Check if solutions have processing time divided by node clock speed"""
import json
import glob

def check_timing_issues(solution_path, app_path):
    """Check if processing times are correctly calculated with clock speed division"""
    
    # Load application
    with open(app_path, 'r') as f:
        app = json.load(f)
    
    # Load solution
    with open(solution_path, 'r') as f:
        solution = json.load(f)
    
    # Get platform info (assume same platform file naming)
    app_name = app_path.split('/')[-1].replace('.json', '')
    
    # Build task processing times
    task_times = {job['id']: job['processing_times'] for job in app['application']['jobs']}
    
    issues = []
    
    for task in solution:
        task_id = task['task_id']
        start = task['start_time']
        end = task['end_time']
        duration = end - start
        
        if task_id in task_times:
            expected_time = task_times[task_id]
            
            # Check if duration is close to raw processing time (not divided by clock speed)
            if abs(duration - expected_time) < 1.0:  # Within 1 second
                issues.append({
                    'task_id': task_id,
                    'node_id': task['node_id'],
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'expected_processing_time': expected_time,
                    'issue': 'Processing time NOT divided by clock speed (duration matches raw processing_times)',
                    'severity': 'HIGH'
                })
            # Check if duration is extremely small (likely divided correctly)
            elif duration < 0.01:  # Less than 10ms - likely divided by clock speed
                # This is probably correct
                pass
            # Check if duration is suspiciously large
            elif duration > expected_time * 2:
                issues.append({
                    'task_id': task_id,
                    'node_id': task['node_id'],
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'expected_processing_time': expected_time,
                    'issue': f'Duration {duration} is much larger than processing_time {expected_time}',
                    'severity': 'MEDIUM'
                })
    
    return issues

def main():
    solutions = sorted(glob.glob('solution/*_ga.json'))[:30]  # Check first 30
    
    print("="*70)
    print("PROCESSING TIME / CLOCK SPEED VERIFICATION")
    print("="*70)
    print(f"Checking {len(solutions)} solutions for timing issues...\n")
    
    total_high = 0
    total_medium = 0
    
    for sol_path in solutions:
        sol_name = sol_path.replace('solution\\', '').replace('solution/', '')
        app_name = sol_name.replace('_ga.json', '.json')
        app_path = f'Application/{app_name}'
        
        try:
            issues = check_timing_issues(sol_path, app_path)
            
            high_issues = [i for i in issues if i['severity'] == 'HIGH']
            medium_issues = [i for i in issues if i['severity'] == 'MEDIUM']
            
            if high_issues:
                print(f"\n!!! {sol_name} - {len(high_issues)} HIGH severity issues:")
                for issue in high_issues[:3]:  # Show first 3
                    print(f"  Task {issue['task_id']}: duration={issue['duration']:.6f}, " +
                          f"expected={issue['expected_processing_time']}, issue: {issue['issue']}")
                total_high += len(high_issues)
                
            if medium_issues:
                print(f"\n! {sol_name} - {len(medium_issues)} MEDIUM severity issues")
                total_medium += len(medium_issues)
                
            if not issues:
                print(f"OK {sol_name}: No timing issues detected")
                
        except Exception as e:
            print(f"ERROR {sol_name}: {e}")
    
    print("\n" + "="*70)
    print(f"SUMMARY:")
    print(f"  HIGH severity (time not divided by clock): {total_high}")
    print(f"  MEDIUM severity (suspicious durations): {total_medium}")
    print("="*70)

if __name__ == '__main__':
    main()
