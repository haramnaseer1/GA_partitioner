"""
Validation Utilities for GA Schedule Constraint Checking - COMPLETE VERSION
=========================================================

This module provides utilities for validating task scheduling solutions
against GA constraints including precedence, non-overlap, eligibility, 
AND timing consistency.

Based on Guide.pdf §10.7 - Tier-based latency model
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import re # Added for clean clock speed parsing

# =====================================================================
# TIER MAPPING AND LATENCY CONSTANTS (FROM Guide.pdf §10.7)
# =====================================================================

# (Tier functions remain unchanged)

def get_tier_from_node_id(node_id: int) -> str:
    """
    Determines the tier (Edge, Fog, Cloud) based on the node ID range.
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
    """
    # Delay matrix (from Guide.pdf §10.7)
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
    """
    return {
        1: 'General purpose CPU',
        2: 'FPGA',
        3: 'Raspberry Pi 5',
        4: 'Microcontroller',
        5: 'High Performance CPU',
        6: 'Graphical Processing Unit'
    }


# =====================================================================
# DATA EXTRACTION HELPERS
# =====================================================================

def parse_clock_speed(speed_str: str) -> float:
    """
    Converts clock speed string (e.g., '1.5 GHz', '300 MHz', '16 MHz') to MHz float.
    Assumes 1 GHz = 1000 MHz.
    """
    match = re.match(r'(\d+\.?\d*)\s*(GHz|MHz)', speed_str, re.IGNORECASE)
    if not match:
        return 0.0
    
    value = float(match.group(1))
    unit = match.group(2).upper()
    
    if unit == 'GHZ':
        return value * 1000.0
    elif unit == 'MHZ':
        return value
    return 0.0


def prepare_processor_details(platform_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Extracts a map of processor Node ID to its Type and Clock Speed (in MHz).
    """
    processor_details = {}
    for node in platform_json['platform']['nodes']:
        if not node.get('is_router', False):
            speed_str = node.get('clocking_speed', '0 MHz')
            processor_details[node['id']] = {
                'type': node.get('type_of_processor'),
                'speed_mhz': parse_clock_speed(speed_str)
            }
    return processor_details


def prepare_task_eligibility(application_json: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Extracts a map of Task ID to allowed Processor Types.
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


def prepare_task_performance(application_json: Dict[str, Any]) -> Dict[int, float]:
    """
    Extracts a map of Task ID to its processing time (WCET).
    """
    task_performance = {}
    for job in application_json['application']['jobs']:
        # Assuming WCET is stored under 'processing_times' or 'wcet_fullspeed'
        # Using 'processing_times' as it appears in T2.json alongside WCET.
        task_performance[job['id']] = float(job.get('processing_times', job.get('wcet_fullspeed', 0)))
    return task_performance


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

def check_timing_consistency(
    solution: List[Dict[str, Any]],
    task_performance: Dict[int, float],
    processor_details: Dict[int, Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Check if the scheduled task duration matches the expected duration based on
    WCET (processing_times) and the assigned processor's clock speed.
    
    Assumes WCET is normalized to a 1000 MHz (1 GHz) reference speed.
    
    Expected_Duration = WCET * (Reference_Speed / Node_Speed)
    """
    violations = []
    REFERENCE_SPEED_MHZ = 1000.0
    TOLERANCE = 0.05  # Allow 5% deviation

    for task in solution:
        task_id = task['task_id']
        node_id = task['node_id']
        
        # 1. Get required data
        wcet = task_performance.get(task_id, 0.0)
        
        if node_id not in processor_details:
             violations.append(f"Timing violation: Node {node_id} details not found in platform.")
             continue
             
        node_speed = processor_details[node_id]['speed_mhz']
        scheduled_duration = task['end_time'] - task['start_time']
        
        if wcet <= 0 or node_speed <= 0:
            # Cannot check consistency if data is missing or zero
            continue
            
        # 2. Calculate expected duration
        # We check for the scaling factor error only if the expected time is significant (e.g., > 10 time units)
        expected_duration = wcet * (REFERENCE_SPEED_MHZ / node_speed)
        
        # 3. Compare with scheduled duration
        if abs(scheduled_duration - expected_duration) / expected_duration > TOLERANCE:
            violations.append(
                f"Timing violation: Task {task_id} on Node {node_id} ({node_speed:.1f} MHz). "
                f"Expected duration: {expected_duration:.4f}, Scheduled: {scheduled_duration:.4f}. "
                f"Deviation: {abs(scheduled_duration - expected_duration) / expected_duration * 100:.2f}%"
            )
            
    return len(violations) == 0, violations


def check_precedence_constraints(
    solution: List[Dict[str, Any]],
    application: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check if precedence constraints are satisfied.
    """
    violations = []
    
    # Build task mapping for quick lookup
    task_map = {task['task_id']: task for task in solution}
    
    for task in solution:
        receiver_id = task['task_id']
        receiver_start = task['start_time']
        dependencies = task.get('dependencies', [])
        
        for dep in dependencies:
            sender_id = dep['task_id']
            # Message_size is assumed to contain the full precedence delay (T_comm)
            precedence_delay = dep.get('message_size', 0) 
            
            if sender_id not in task_map:
                violations.append(
                    f"Dependency error: sender task {sender_id} not found for receiver {receiver_id}"
                )
                continue
            
            sender_task = task_map[sender_id]
            sender_end = sender_task['end_time']
            
            # Check: Start(receiver) >= End(sender) + precedence_delay
            required_start = sender_end + precedence_delay
            
            # Use a small epsilon for floating point comparison tolerance
            EPSILON = 1e-6 
            
            if receiver_start < required_start - EPSILON:
                actual_gap = receiver_start - sender_end
                violations.append(
                    f"Precedence violation: Task {sender_id} -> {receiver_id}, "
                    f"Required start: {required_start:.4f}, Actual start: {receiver_start:.4f}. "
                    f"Required delay={precedence_delay:.4f}, actual gap={actual_gap:.4f}"
                )
    
    return len(violations) == 0, violations


def check_non_overlap_constraints(
    solution: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Check if tasks on the same processor have non-overlapping execution windows.
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
            
            # Use a small epsilon for floating point comparison tolerance
            EPSILON = 1e-6 
            
            # Only flag if tasks truly overlap (next_start is before curr_end)
            if next_start < curr_end - EPSILON:
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
    """
    violations = []
    
    # Get processor types
    processor_details = prepare_processor_details(platform)
    processor_types = {k: v['type'] for k, v in processor_details.items()}
    
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
                f"Eligibility violation: Task {task_id} assigned to Node {node_id} ({actual_type}), "
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
    Comprehensive validation of a GA solution, including timing consistency.
    """
    # Load data
    solution = load_solution_data(solution_path)
    application = load_application_data(application_path)
    platform = load_platform_data(platform_path)
    
    # Prepare lookup data for complex checks
    task_performance = prepare_task_performance(application)
    processor_details = prepare_processor_details(platform)
    
    # Check constraints
    precedence_ok, precedence_violations = check_precedence_constraints(solution, application)
    overlap_ok, overlap_violations = check_non_overlap_constraints(solution)
    eligibility_ok, eligibility_violations = check_eligibility_constraints(
        solution, application, platform
    )
    timing_ok, timing_violations = check_timing_consistency(
        solution, task_performance, processor_details
    )
    
    # Calculate makespan
    makespan = max([task.get('end_time', 0) for task in solution]) if solution else 0
    
    results = {
        'valid': precedence_ok and overlap_ok and eligibility_ok and timing_ok,
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
        },
        'timing_consistency': {
            'satisfied': timing_ok,
            'violations': timing_violations
        }
    }
    
    if verbose:
        print("="*70)
        print(f"VALIDATION RESULTS for {solution_path}")
        print("="*70)
        print(f"Valid: {'YES' if results['valid'] else 'NO'}")
        print(f"Makespan: {makespan:.4f}")
        
        # --- Print details of checks ---
        checks = [
            ('Precedence Constraints', precedence_ok, precedence_violations),
            ('Non-Overlap Constraints', overlap_ok, overlap_violations),
            ('Eligibility Constraints', eligibility_ok, eligibility_violations),
            ('Timing Consistency', timing_ok, timing_violations)
        ]
        
        for name, ok, violations in checks:
            print(f"\n{name}: {'PASS' if ok else 'FAIL'}")
            if violations:
                for v in violations[:5]: 
                    print(f"  - {v}")
                if len(violations) > 5:
                    print(f"  ... and {len(violations) - 5} more")
        
        print("="*70)
    
    return results