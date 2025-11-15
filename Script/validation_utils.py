"""
Validation Utilities for GA Schedule Constraint Checking
=========================================================

This module provides utilities for validating task scheduling solutions
against GA constraints including precedence, non-overlap, and eligibility.

Based on Guide.pdf §10.7 - Tier-based latency model
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


# =====================================================================
# TIER MAPPING AND LATENCY CONSTANTS (FROM Guide.pdf §10.7)
# =====================================================================

def get_tier_from_node_id(node_id: int) -> str:
    """
    Determines the tier (Edge, Fog, Cloud) based on the node ID range.
    
    Tier ranges:
    - Edge: 11-99
    - Fog: 101-999
    - Cloud: ≥1001
    
    Args:
        node_id: Platform node ID
        
    Returns:
        str: 'Edge', 'Fog', 'Cloud', or 'Unknown'
    """
    if 11 <= node_id <= 99:
        return 'Edge'
    elif 101 <= node_id <= 999:
        return 'Fog'
    elif node_id >= 1001:
        return 'Cloud'
    else:
        return 'Unknown'


def get_inter_tier_delay(sender_tier: str, receiver_tier: str) -> float:
    """
    Returns the fixed tier-based communication delay constant.
    
    Delay matrix (from Guide.pdf §10.7):
    - Edge ↔ Edge: 50
    - Edge ↔ Fog: 150
    - Edge ↔ Cloud: 350
    - Fog ↔ Cloud: 200
    - Cloud ↔ Cloud: 175
    - Same node: 0
    
    Args:
        sender_tier: Tier of sending task
        receiver_tier: Tier of receiving task
        
    Returns:
        float: Communication delay in time units
    """
    # Tier-based delay constants
    DELAYS = {
        ('Edge', 'Edge'): 50.0,
        ('Edge', 'Fog'): 150.0,
        ('Fog', 'Edge'): 150.0,
        ('Edge', 'Cloud'): 350.0,
        ('Cloud', 'Edge'): 350.0,
        ('Fog', 'Cloud'): 200.0,
        ('Cloud', 'Fog'): 200.0,
        ('Cloud', 'Cloud'): 175.0,
        ('Fog', 'Fog'): 50.0,  # Assuming same as Edge-Edge for intra-tier
    }
    
    # Same tier, different nodes
    if sender_tier == receiver_tier:
        return DELAYS.get((sender_tier, receiver_tier), 0.0)
    
    return DELAYS.get((sender_tier, receiver_tier), 0.0)


def calculate_precedence_delay(
    sender_node_id: int,
    receiver_node_id: int,
    message_size: float
) -> float:
    """
    Calculates the minimum required delay (Δ) between a sender's end time
    and a receiver's start time.
    
    In the GA implementation, the message_size already includes the path_cost
    (communication delay), so the precedence delay is simply the message_size.
    
    Args:
        sender_node_id: Node ID where sender task executes
        receiver_node_id: Node ID where receiver task executes
        message_size: Size of message (already includes path_cost from GA)
        
    Returns:
        float: Minimum precedence delay (just the message_size)
    """
    # The GA adds path_cost to message_size at line 309 of global_GA.py
    # So message_size already represents the full communication delay
    return message_size


# =====================================================================
# RESOURCE MAPPING
# =====================================================================

def resource_mapping() -> Dict[int, str]:
    """
    Maps GA resource class IDs to processor type strings.
    
    Returns:
        dict: Resource ID → Processor type name
    """
    return {
        1: 'General purpose CPU',
        2: 'FPGA',
        3: 'Raspberry Pi 5',
        4: 'Microcontroller',
        5: 'High Performance CPU',
        6: 'GPU'
    }


# =====================================================================
# DATA EXTRACTION HELPERS
# =====================================================================

def prepare_processor_eligibility(platform_json: Dict[str, Any]) -> Dict[int, str]:
    """
    Extracts a map of processor Node ID to its Type.
    
    Args:
        platform_json: Platform configuration dictionary
        
    Returns:
        dict: Node ID → Processor type
    """
    processor_types = {}
    for node in platform_json['platform']['nodes']:
        if not node.get('is_router', False):
            processor_types[node['id']] = node.get('type_of_processor')
    return processor_types


def prepare_task_eligibility(application_json: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Extracts a map of Task ID to allowed Processor Types.
    
    Args:
        application_json: Application graph dictionary
        
    Returns:
        dict: Task ID → List of allowed processor types
    """
    task_eligibility = {}
    resource_map = resource_mapping()
    
    for job in application_json['application']['jobs']:
        allowed_resource_ids = job.get('can_run_on', [])
        allowed_types = [
            resource_map.get(res_id) 
            for res_id in allowed_resource_ids 
            if resource_map.get(res_id)
        ]
        task_eligibility[job['id']] = allowed_types
    
    return task_eligibility


def load_application_data(app_path: str) -> Dict[str, Any]:
    """Load application JSON file."""
    with open(app_path, 'r') as f:
        return json.load(f)


def load_platform_data(platform_path: str) -> Dict[str, Any]:
    """Load platform JSON file."""
    with open(platform_path, 'r') as f:
        return json.load(f)


def load_solution_data(solution_path: str) -> List[Dict[str, Any]]:
    """Load GA solution JSON file."""
    with open(solution_path, 'r') as f:
        data = json.load(f)
        # Handle both formats: list directly or dict with 'solution' key
        if isinstance(data, list):
            return data
        return data.get('solution', [])


# =====================================================================
# CONSTRAINT VALIDATION FUNCTIONS
# =====================================================================

def check_precedence_constraints(
    solution: List[Dict[str, Any]],
    application: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check if precedence constraints are satisfied.
    
    For each dependency u → v:
    Start(v) ≥ End(u) + Δ(u,v)
    
    Args:
        solution: List of task assignments with timing and dependencies
        application: Application graph (not used, kept for compatibility)
        
    Returns:
        tuple: (all_satisfied, list of violations)
    """
    violations = []
    
    # Build task mapping for quick lookup
    task_map = {task['task_id']: task for task in solution}
    
    # Check dependencies for each task using solution's dependency information
    for task in solution:
        receiver_id = task['task_id']
        receiver_start = task['start_time']
        dependencies = task.get('dependencies', [])
        
        for dep in dependencies:
            sender_id = dep['task_id']
            message_size = dep.get('message_size', 0)
            
            if sender_id not in task_map:
                violations.append(
                    f"Dependency error: sender task {sender_id} not found for receiver {receiver_id}"
                )
                continue
            
            sender_task = task_map[sender_id]
            sender_end = sender_task['end_time']
            
            # Check: Start(receiver) >= End(sender) + message_size
            required_start = sender_end + message_size
            actual_gap = receiver_start - sender_end
            
            if receiver_start < required_start - 1e-6:  # Small epsilon for floating point
                violations.append(
                    f"Precedence violation: Task {sender_id} -> {receiver_id}, "
                    f"required delay={message_size:.2f}, actual gap={actual_gap:.2f}"
                )
    
    return len(violations) == 0, violations


def check_non_overlap_constraints(
    solution: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Check if tasks on the same processor have non-overlapping execution windows.
    Tasks can start exactly when another ends (non-strict inequality).
    
    Args:
        solution: List of task assignments with timing
        
    Returns:
        tuple: (all_satisfied, list of violations)
    """
    violations = []
    
    # Group tasks by processor
    processor_tasks = {}
    for task in solution:
        node_id = task['node_id']
        if node_id not in processor_tasks:
            processor_tasks[node_id] = []
        processor_tasks[node_id].append(task)
    
    # Check each processor
    for node_id, tasks in processor_tasks.items():
        # Sort by start time
        tasks_sorted = sorted(tasks, key=lambda t: t.get('start_time', 0))
        
        # Check consecutive pairs - strict overlap only (not boundary touching)
        for i in range(len(tasks_sorted) - 1):
            curr = tasks_sorted[i]
            next_task = tasks_sorted[i + 1]
            
            curr_end = curr.get('end_time', 0)
            next_start = next_task.get('start_time', 0)
            
            # Only flag if tasks truly overlap (one starts before the other ends)
            # Allow exact boundary matching (curr_end == next_start)
            if next_start < curr_end - 1e-6:  # True overlap
                violations.append(
                    f"Overlap on processor {node_id}: "
                    f"Task {curr['task_id']} (ends {curr_end:.6f}) overlaps with "
                    f"Task {next_task['task_id']} (starts {next_start:.6f})"
                )
    
    return len(violations) == 0, violations


def check_eligibility_constraints(
    solution: List[Dict[str, Any]],
    application: Dict[str, Any],
    platform: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check if tasks are assigned to eligible processors.
    
    Args:
        solution: List of task assignments
        application: Application graph with task requirements
        platform: Platform configuration
        
    Returns:
        tuple: (all_satisfied, list of violations)
    """
    violations = []
    
    # Get processor types
    processor_types = prepare_processor_eligibility(platform)
    
    # Get task eligibility
    task_eligibility = prepare_task_eligibility(application)
    
    # Check each assignment
    for task in solution:
        task_id = task['task_id']
        node_id = task['node_id']
        
        # Get actual processor type
        actual_type = processor_types.get(node_id, 'Unknown')
        
        # Get allowed types
        allowed_types = task_eligibility.get(task_id, [])
        
        if actual_type not in allowed_types:
            violations.append(
                f"Eligibility violation: Task {task_id} assigned to {actual_type}, "
                f"but can only run on {allowed_types}"
            )
    
    return len(violations) == 0, violations


def validate_solution(
    solution_path: str,
    application_path: str,
    platform_path: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive validation of a GA solution.
    
    Args:
        solution_path: Path to solution JSON
        application_path: Path to application JSON
        platform_path: Path to platform JSON
        verbose: Print detailed results
        
    Returns:
        dict: Validation results with all constraint checks
    """
    # Load data
    solution = load_solution_data(solution_path)
    application = load_application_data(application_path)
    platform = load_platform_data(platform_path)
    
    # Check constraints
    precedence_ok, precedence_violations = check_precedence_constraints(solution, application)
    overlap_ok, overlap_violations = check_non_overlap_constraints(solution)
    eligibility_ok, eligibility_violations = check_eligibility_constraints(
        solution, application, platform
    )
    
    # Calculate makespan
    makespan = max([task.get('end_time', 0) for task in solution]) if solution else 0
    
    results = {
        'valid': precedence_ok and overlap_ok and eligibility_ok,
        'makespan': makespan,
        'precedence': {
            'satisfied': precedence_ok,
            'violations': precedence_violations
        },
        'non_overlap': {
            'satisfied': overlap_ok,
            'violations': overlap_violations
        },
        'eligibility': {
            'satisfied': eligibility_ok,
            'violations': eligibility_violations
        }
    }
    
    if verbose:
        print("="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        print(f"Solution: {solution_path}")
        print(f"Valid: {'YES' if results['valid'] else 'NO'}")
        print(f"Makespan: {makespan:.2f}")
        print(f"\nPrecedence Constraints: {'PASS' if precedence_ok else 'FAIL'}")
        if precedence_violations:
            for v in precedence_violations[:5]:  # Show first 5
                print(f"  - {v}")
            if len(precedence_violations) > 5:
                print(f"  ... and {len(precedence_violations) - 5} more")
        
        print(f"\nNon-Overlap Constraints: {'PASS' if overlap_ok else 'FAIL'}")
        if overlap_violations:
            for v in overlap_violations[:5]:
                print(f"  - {v}")
            if len(overlap_violations) > 5:
                print(f"  ... and {len(overlap_violations) - 5} more")
        
        print(f"\nEligibility Constraints: {'PASS' if eligibility_ok else 'FAIL'}")
        if eligibility_violations:
            for v in eligibility_violations[:5]:
                print(f"  - {v}")
            if len(eligibility_violations) > 5:
                print(f"  ... and {len(eligibility_violations) - 5} more")
        print("="*70)
    
    return results
