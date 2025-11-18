"""
GA Solution Validation Script
==============================

Command-line tool for validating GA scheduling solutions against constraints.
...
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Script.validation_utils import validate_solution


def find_application_file(solution_name: str, app_dir: str = 'Application') -> str:
# ... (find_application_file remains unchanged)
    # Remove _ga.json suffix
    base_name = solution_name.replace('_ga.json', '')
    
    # Remove seed suffix if present (e.g., _seed00)
    import re
    base_name = re.sub(r'_seed\d+$', '', base_name)
    
    # Add .json extension
    base_name = base_name + '.json'
    
    # Get script directory and construct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    app_path = os.path.join(project_dir, app_dir, base_name)
    
    if not os.path.exists(app_path):
        # Try without extension change
        alt_name = solution_name.replace('_ga', '')
        app_path = os.path.join(project_dir, app_dir, alt_name)
    
    return app_path


def find_platform_file(application_path: str = None, platform_dir: str = 'Platform') -> str:
    """
    Find the platform configuration file.
    **FIXED: Forces T2 applications to use the comprehensive 3_Tier_Platform.**
    
    Args:
        application_path: Path to application file (used to detect platform number)
        platform_dir: Directory containing platform files
        
    Returns:
        str: Path to platform file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    platform_name = '5_Platform.json'  # default fallback
    
    if application_path:
        app_basename = os.path.basename(application_path)
        
        # --- FIX FOR T2_VAR APPLICATIONS ---
        if app_basename.startswith('T2'):
             # T2 solutions use nodes (53, 54) found in 3_Tier_Platform.json or 5_Platform.json
             platform_name = '3_Tier_Platform.json' 
             platform_path = os.path.join(project_dir, platform_dir, platform_name)
             
             if os.path.exists(platform_path):
                 return platform_path
             # Fallback to the other comprehensive platform
             platform_name = '5_Platform.json'
        # -----------------------------------
        
        # Original Logic: Extract number from T#_var format
        import re
        match = re.match(r'[Tt](\d+)_', app_basename)  # Match T#_var format only
        if match:
            platform_num = match.group(1)
            platform_name = f'{platform_num}_Platform.json'
    
    platform_path = os.path.join(project_dir, platform_dir, platform_name)
    
    # Fallback if platform doesn't exist
    if not os.path.exists(platform_path):
        platform_name = 'EdgeAI-Trust_Platform.json'
        platform_path = os.path.join(project_dir, platform_dir, platform_name)
    
    return platform_path


def validate_single_solution(
# ... (validate_single_solution remains unchanged)
    solution_path: str,
    application_path: str = None,
    platform_path: str = None,
    verbose: bool = True
) -> dict:
    """
    Validate a single solution file.
    
    Args:
        solution_path: Path to solution JSON file
        application_path: Path to application JSON (auto-detected if None)
        platform_path: Path to platform JSON (auto-detected if None)
        verbose: Print detailed results
        
    Returns:
        dict: Validation results
    """
    # Auto-detect application file
    if application_path is None:
        solution_name = os.path.basename(solution_path)
        application_path = find_application_file(solution_name)
    
    # Auto-detect platform file
    if platform_path is None:
        platform_path = find_platform_file(application_path)
    
    # Verify files exist
    if not os.path.exists(solution_path):
        raise FileNotFoundError(f"Solution file not found: {solution_path}")
    if not os.path.exists(application_path):
        raise FileNotFoundError(f"Application file not found: {application_path}")
    if not os.path.exists(platform_path):
        raise FileNotFoundError(f"Platform file not found: {platform_path}")
    
    # Validate
    return validate_solution(solution_path, application_path, platform_path, verbose=verbose)


def validate_all_solutions(
# ... (validate_all_solutions remains unchanged)
    solution_dir: str = 'solution',
    output_file: str = None,
    verbose: bool = False
) -> dict:
    """
    Validate all solution files in a directory.
    
    Args:
        solution_dir: Directory containing solution files
        output_file: Optional path to save results JSON
        verbose: Print detailed results for each solution
        
    Returns:
        dict: Summary statistics
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    sol_dir_path = os.path.join(project_dir, solution_dir)
    
    # Find all solution files
    solution_files = [
        os.path.join(sol_dir_path, f)
        for f in os.listdir(sol_dir_path)
        if f.endswith('_ga.json')
    ]
    
    print(f"\n{'='*70}")
    print(f"Validating {len(solution_files)} solutions from {solution_dir}/")
    print(f"{'='*70}\n")
    
    results = {}
    valid_count = 0
    total_makespan = 0.0
    
    for sol_path in sorted(solution_files):
        sol_name = os.path.basename(sol_path)
        
        try:
            result = validate_single_solution(sol_path, verbose=verbose)
            results[sol_name] = result
            
            if result['valid']:
                valid_count += 1
            
            total_makespan += result['makespan']
            
            # Print summary line
            status = "✅ VALID" if result['valid'] else "❌ INVALID"
            print(f"{sol_name:40s} {status:12s} Makespan: {result['makespan']:8.2f}")
            
        except Exception as e:
            print(f"{sol_name:40s} ❌ ERROR: {str(e)}")
            results[sol_name] = {'error': str(e)}
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total solutions:     {len(solution_files)}")
    print(f"Valid solutions:     {valid_count} ({100*valid_count/len(solution_files):.1f}%)")
    print(f"Invalid solutions:   {len(solution_files) - valid_count}")
    print(f"Average makespan:    {total_makespan/len(solution_files):.2f}")
    print(f"{'='*70}\n")
    
    # Save results if requested
    if output_file:
        output_path = os.path.join(project_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {output_file}")
    
    return {
        'total': len(solution_files),
        'valid': valid_count,
        'invalid': len(solution_files) - valid_count,
        'average_makespan': total_makespan / len(solution_files) if solution_files else 0,
        'results': results
    }


def main():
# ... (main remains unchanged)
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(
        description='Validate GA scheduling solutions against constraints'
    )
    
    parser.add_argument(
        '--solution',
        type=str,
        help='Path to single solution file to validate'
    )
    
    parser.add_argument(
        '--application',
        type=str,
        default=None,
        help='Path to application file (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--platform',
        type=str,
        default=None,
        help='Path to platform file (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all solutions in solution/ directory'
    )
    
    parser.add_argument(
        '--solution_dir',
        type=str,
        default='solution',
        help='Directory containing solutions (for --all mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save validation results to JSON file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed validation results'
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Validate all solutions
        validate_all_solutions(
            solution_dir=args.solution_dir,
            output_file=args.output,
            verbose=args.verbose
        )
    elif args.solution:
        # Validate single solution
        validate_single_solution(
            solution_path=args.solution,
            application_path=args.application,
            platform_path=args.platform,
            verbose=True
        )
    else:
        parser.print_help()
        print("\nError: Either --solution or --all must be specified")
        sys.exit(1)


if __name__ == "__main__":
    main()