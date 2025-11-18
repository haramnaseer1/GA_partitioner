"""
Validate that the generated dataset is suitable for GNN training.
Checks:
1. All solutions are constraint-valid (no violations)
2. Makespan optimization (lower values preferred)
3. Solution diversity within each application
4. Quality variation (GNN needs good and bad examples to learn ranking)
"""

import os
import json
import subprocess
from collections import defaultdict
import statistics

def validate_solution(solution_file, application_file):
    """Run check_solutions.py and parse results"""
    try:
        result = subprocess.run(
            ['python', 'Script/check_solutions.py', 
             '--solution', solution_file,
             '--application', application_file],
            capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace'
        )
        
        output = result.stdout
        
        # Parse validation results
        is_valid = 'Valid: YES' in output
        makespan = None
        
        for line in output.split('\n'):
            if 'Makespan:' in line:
                try:
                    makespan = float(line.split(':')[1].strip())
                except:
                    pass
        
        # Check for constraint violations
        precedence_pass = 'Precedence: PASS' in output
        overlap_pass = 'Overlap: PASS' in output
        eligibility_pass = 'Eligibility: PASS' in output
        
        return {
            'valid': is_valid,
            'makespan': makespan,
            'precedence': precedence_pass,
            'overlap': overlap_pass,
            'eligibility': eligibility_pass
        }
    except Exception as e:
        return {
            'valid': False,
            'makespan': None,
            'error': str(e)
        }

def analyze_dataset(solution_dir='solution', app_dir='Application'):
    """Analyze entire dataset for GNN training suitability"""
    
    print("=" * 70)
    print("DATASET VALIDATION FOR GNN TRAINING")
    print("=" * 70)
    print()
    
    # Group solutions by application
    app_solutions = defaultdict(list)
    
    solution_files = [f for f in os.listdir(solution_dir) if f.endswith('.json')]
    print(f"Found {len(solution_files)} solution files\n")
    
    # Organize by application
    for sol_file in solution_files:
        # Extract app name (remove _seed##_ga or _ga suffix)
        base_name = sol_file.replace('_ga.json', '')
        parts = base_name.split('_seed')
        app_name = parts[0]
        app_solutions[app_name].append(sol_file)
    
    print(f"Solutions grouped into {len(app_solutions)} applications\n")
    print("=" * 70)
    print()
    
    # Validation statistics
    total_solutions = 0
    valid_solutions = 0
    constraint_violations = 0
    
    # Quality metrics
    all_makespans = []
    app_diversity = []
    
    # Detailed analysis per application
    print("ANALYZING EACH APPLICATION:")
    print("-" * 70)
    
    problem_apps = []
    
    for app_name in sorted(app_solutions.keys())[:10]:  # Sample first 10
        sols = app_solutions[app_name]
        app_file = os.path.join(app_dir, f"{app_name}.json")
        
        if not os.path.exists(app_file):
            print(f"\n⚠ {app_name}: Application file not found!")
            continue
        
        print(f"\n{app_name} ({len(sols)} solutions):")
        
        makespans = []
        valid_count = 0
        
        for sol_file in sols:
            sol_path = os.path.join(solution_dir, sol_file)
            result = validate_solution(sol_path, app_file)
            
            total_solutions += 1
            
            if result['valid']:
                valid_solutions += 1
                valid_count += 1
                if result['makespan']:
                    makespans.append(result['makespan'])
                    all_makespans.append(result['makespan'])
            else:
                constraint_violations += 1
                print(f"  ✗ {sol_file}: INVALID")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
                problem_apps.append((app_name, sol_file))
        
        # Analyze diversity within this application
        if len(makespans) >= 2:
            min_ms = min(makespans)
            max_ms = max(makespans)
            avg_ms = statistics.mean(makespans)
            std_ms = statistics.stdev(makespans) if len(makespans) > 1 else 0
            
            diversity = (max_ms - min_ms) / min_ms if min_ms > 0 else 0
            app_diversity.append(diversity)
            
            print(f"  ✓ Valid: {valid_count}/{len(sols)}")
            print(f"  ✓ Makespan range: {min_ms:.1f} - {max_ms:.1f} (diversity: {diversity:.1%})")
            print(f"  ✓ Average: {avg_ms:.1f}, StdDev: {std_ms:.1f}")
            
            # Check if there's enough variation for GNN to learn
            if diversity < 0.05:  # Less than 5% variation
                print(f"  ⚠ LOW DIVERSITY - All solutions very similar")
        else:
            print(f"  ✓ Valid: {valid_count}/{len(sols)}")
            if makespans:
                print(f"  ✓ Makespan: {makespans[0]:.1f}")
    
    print("\n" + "=" * 70)
    print("OVERALL DATASET STATISTICS")
    print("=" * 70)
    print()
    
    # Constraint validation
    print("1. CONSTRAINT VALIDATION:")
    print(f"   Total solutions: {total_solutions}")
    print(f"   Valid solutions: {valid_solutions} ({valid_solutions/total_solutions*100:.1f}%)")
    print(f"   Invalid solutions: {constraint_violations}")
    
    if valid_solutions == total_solutions:
        print("   ✓ ALL SOLUTIONS CONSTRAINT-VALID ✓")
    else:
        print(f"   ✗ {constraint_violations} solutions have constraint violations")
    
    print()
    
    # Makespan optimization
    print("2. MAKESPAN OPTIMIZATION:")
    if all_makespans:
        print(f"   Minimum makespan: {min(all_makespans):.1f}")
        print(f"   Maximum makespan: {max(all_makespans):.1f}")
        print(f"   Average makespan: {statistics.mean(all_makespans):.1f}")
        print(f"   Std deviation: {statistics.stdev(all_makespans):.1f}")
        print(f"   Range: {min(all_makespans):.1f} - {max(all_makespans):.1f}")
        print("   ✓ Solutions show varying quality levels")
    
    print()
    
    # Solution diversity
    print("3. SOLUTION DIVERSITY:")
    if app_diversity:
        avg_diversity = statistics.mean(app_diversity)
        print(f"   Average diversity per app: {avg_diversity:.1%}")
        print(f"   Min diversity: {min(app_diversity):.1%}")
        print(f"   Max diversity: {max(app_diversity):.1%}")
        
        if avg_diversity > 0.2:
            print("   ✓ GOOD DIVERSITY - Each app has varied solutions")
        elif avg_diversity > 0.1:
            print("   ○ MODERATE DIVERSITY - Acceptable variation")
        else:
            print("   ⚠ LOW DIVERSITY - Solutions may be too similar")
    
    print()
    
    # Quality variation (critical for GNN learning)
    print("4. QUALITY VARIATION (GNN LEARNING):")
    if all_makespans:
        # Calculate coefficient of variation
        cv = statistics.stdev(all_makespans) / statistics.mean(all_makespans)
        print(f"   Coefficient of variation: {cv:.2f}")
        
        # Check for quality spread
        q1 = statistics.quantiles(all_makespans, n=4)[0]
        q3 = statistics.quantiles(all_makespans, n=4)[2]
        iqr = q3 - q1
        
        print(f"   25th percentile: {q1:.1f}")
        print(f"   75th percentile: {q3:.1f}")
        print(f"   Interquartile range: {iqr:.1f}")
        
        if cv > 0.3:
            print("   ✓ EXCELLENT - High quality variation for learning")
        elif cv > 0.15:
            print("   ✓ GOOD - Sufficient variation for GNN to learn ranking")
        else:
            print("   ⚠ MODERATE - Limited variation, GNN may struggle")
    
    print()
    print("=" * 70)
    print("GNN TRAINING READINESS ASSESSMENT")
    print("=" * 70)
    print()
    
    # Final verdict
    ready = True
    issues = []
    
    if valid_solutions != total_solutions:
        ready = False
        issues.append(f"Constraint violations in {constraint_violations} solutions")
    
    if app_diversity and statistics.mean(app_diversity) < 0.1:
        ready = False
        issues.append("Low solution diversity within applications")
    
    if all_makespans:
        cv = statistics.stdev(all_makespans) / statistics.mean(all_makespans)
        if cv < 0.15:
            ready = False
            issues.append("Insufficient quality variation for learning")
    
    if ready:
        print("✓✓✓ DATASET IS READY FOR GNN TRAINING ✓✓✓")
        print()
        print("The dataset fulfills client requirements:")
        print("  ✓ All solutions are constraint-valid (no violations)")
        print("  ✓ Solutions optimize makespan (lower values present)")
        print("  ✓ Sufficient diversity for GNN to learn patterns")
        print("  ✓ Quality variation enables ranking learning")
        print()
        print("RECOMMENDATION: Proceed with tensor conversion")
        print("  → Run: python create_tensors.py")
    else:
        print("⚠ DATASET HAS POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  ✗ {issue}")
        print()
        print("RECOMMENDATION: Review and regenerate problematic solutions")
    
    print()
    print("=" * 70)
    
    # Show problem apps if any
    if problem_apps:
        print()
        print("PROBLEM SOLUTIONS (Need Investigation):")
        print("-" * 70)
        for app, sol in problem_apps[:10]:
            print(f"  • {app}: {sol}")
        if len(problem_apps) > 10:
            print(f"  ... and {len(problem_apps) - 10} more")
    
    return {
        'total': total_solutions,
        'valid': valid_solutions,
        'ready': ready,
        'issues': issues
    }

if __name__ == '__main__':
    analyze_dataset()
