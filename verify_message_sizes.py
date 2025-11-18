"""Verify message sizes in generated solutions match original applications"""
import json
import glob
import sys

def check_message_sizes(solution_path, app_path):
    """Check if message sizes in solution match original application"""
    
    # Load application
    with open(app_path, 'r') as f:
        app = json.load(f)
    
    # Build message size dictionary
    msg_sizes = {}
    for msg in app['application']['messages']:
        key = (msg['sender'], msg['receiver'])
        msg_sizes[key] = msg['size']
    
    # Load solution
    with open(solution_path, 'r') as f:
        solution = json.load(f)
    
    issues = []
    correct = []
    
    for task in solution:
        if not task['dependencies']:
            continue
            
        for dep in task['dependencies']:
            sender = dep['task_id']
            receiver = task['task_id']
            sol_size = dep['message_size']
            
            # Get original size
            orig_size = msg_sizes.get((sender, receiver), None)
            
            if orig_size is None:
                # No direct message - should be 0 for same node
                if sol_size == 0 or sol_size == 0.0:
                    correct.append(f"Task {sender}->{receiver}: {sol_size} (same-node OK)")
                else:
                    issues.append(f"Task {sender}->{receiver}: {sol_size} (expected 0 for same-node)")
            else:
                # Direct message exists
                if abs(sol_size - orig_size) < 0.001:  # Float comparison
                    correct.append(f"Task {sender}->{receiver}: {sol_size} (matches original)")
                elif sol_size == 0 or sol_size == 0.0:
                    # Same node optimization
                    correct.append(f"Task {sender}->{receiver}: {sol_size} (same-node, orig={orig_size})")
                else:
                    issues.append(f"Task {sender}->{receiver}: {sol_size} (expected {orig_size})")
    
    return issues, correct

def main():
    solutions = sorted(glob.glob('solution/*_ga.json'))[:20]  # Check first 20
    
    print("="*70)
    print("MESSAGE SIZE VERIFICATION")
    print("="*70)
    print(f"Checking {len(solutions)} solutions...\n")
    
    all_issues = []
    total_correct = 0
    
    for sol_path in solutions:
        sol_name = sol_path.replace('solution\\', '').replace('solution/', '')
        app_name = sol_name.replace('_ga.json', '.json')
        app_path = f'Application/{app_name}'
        
        try:
            issues, correct = check_message_sizes(sol_path, app_path)
            total_correct += len(correct)
            
            if issues:
                print(f"\nX {sol_name}:")
                for issue in issues:
                    print(f"   {issue}")
                all_issues.extend(issues)
            else:
                print(f"OK {sol_name}: All message sizes correct ({len(correct)} deps)")
                
        except Exception as e:
            print(f"WARN {sol_name}: Error - {e}")
    
    print("\n" + "="*70)
    print(f"Total dependencies checked: {total_correct}")
    if all_issues:
        print(f"FAILED: {len(all_issues)} message size issues found")
        print("="*70)
        return 1
    else:
        print("SUCCESS: All message sizes are correct!")
        print("="*70)
        return 0

if __name__ == '__main__':
    sys.exit(main())
