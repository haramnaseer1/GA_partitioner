# ******************** List Scheduling ********************
# This program is a list scheduling algorithm for the task alocation to the processors
# The input is a file with the tasks and the processors (Json file which contains a list of tasks and a list of processors)
# The output is a file with the tasks allocated to the processors
# **********************************************************


# Importing the libraries
import json
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sys
from . import config as cfg
import re
import os

# Set the Reference Speed for WCET scaling (1 GHz = 1e9 Hz)
REFERENCE_SPEED_HZ = 1e9

# *************************** Functions ***************************

# Function to convert clocking speed to Hz (Copied from auxiliary_fun_GA.py for self-sufficiency)
def convert_clocking_speed_to_hz(clocking_speed):
    if 'GHz' in clocking_speed:
        # Convert GHz to Hz (1 GHz = 1e9 Hz)
        return float(re.findall(r"\d+\.\d+|\d+", clocking_speed)[0]) * 1e9
    elif 'MHz' in clocking_speed:
        # Convert MHz to Hz (1 MHz = 1e6 Hz)
        return float(re.findall(r"\d+\.\d+|\d+", clocking_speed)[0]) * 1e6
    else:
        return None

# Function to read the Platform Model from the JSON file 
def read_platform_model():
    # Detect platform number from application filename (e.g., T2_var_001 -> 2, T5_var_010 -> 5)
    import re
    import os
    app_name = cfg.file_name  # e.g., "T2_var_001.json" or "T20.json"
    match = re.match(r'[Tt](\d+)_', app_name)  # Match T#_var format only
    if match:
        platform_num = match.group(1)
        platform_file = os.path.join(cfg.platform_dir_path, f"{platform_num}_Platform.json")
    else:
        # Fallback for non-standard names (T20.json, TNC100.json) - use Platform 5
        platform_file = os.path.join(cfg.platform_dir_path, "5_Platform.json")
        if not os.path.exists(platform_file):
            platform_file = os.path.join(cfg.platform_dir_path, "3_Platform.json")
    
    # Add error handling with helpful message
    if not os.path.exists(platform_file):
        raise FileNotFoundError(
            f"Platform file not found: {platform_file}\n"
            f"  Application: {app_name}\n"
            f"  Expected platform: {platform_num if match else '5 (fallback)'}\n"
            f"  platform_dir_path: {cfg.platform_dir_path}\n"
            f"  Current working directory: {os.getcwd()}"
        )
    
    with open(platform_file) as pf:
        pltf_model = json.load(pf)
    PMl = pltf_model['platform']
    return PMl

# Function to extract the Communication Cost from the Application model.
def communication_costs_task(json_data):
    messages = json_data['application']['messages']
    communication_costs = {}

    for message in messages:
        sender = message['sender']
        receiver = message['receiver']
        size = message['size']
        communication_costs[(sender,receiver)] = size

    return communication_costs

def finding_ProcessTime(APP_MODEL):
    #  Function to find the the processing time  for each job
    
    jobs = APP_MODEL['application']['jobs']
   
    # Extract the processing values
    proc_time = {}
    for job in jobs:
        job_id = job['id']
        proc_time[job_id] = job['processing_times']  # extracting the processing time for each job
    
    return proc_time  # returning the task_dag and the processing time for each job

# Function to find the processors and their clock speeds in the platform model
def find_processors(PM):
    processors = []
    clock_speeds = {}
    for node in PM['nodes']:
        if node['is_router'] == False:
            processors.append(node['id'])
            # Extract clock speed and convert to Hz
            clk_speed_str = node.get('clocking_speed', '1 GHz')
            clk_speed_hz = convert_clocking_speed_to_hz(clk_speed_str)
            # Store speed in Hz, falling back to 1 GHz if parsing fails
            clock_speeds[node['id']] = clk_speed_hz if clk_speed_hz else 1e9 
    return processors, clock_speeds # Return list of processors AND their speeds


# ********************* List Scheduling Algorithm *********************


# Function to calculate earliest start time for each task
def calculate_earliest_start_time(graph, processing_time, communication_costs, clock_speeds):
    """
    Calculates the earliest start time (EST) for each task based on a topological sort (HEFT-style).
    This version correctly scales WCET by processor speed.
    """
    earliest_start_time = {}
    
    # Use a dummy "entry" node to simplify calculations for tasks with no predecessors
    for task in nx.topological_sort(graph):
        # For each task, find the maximum time it would take for a predecessor to finish and send its data
        max_predecessor_finish_time = 0
        
        for pred in graph.predecessors(task):
            # Get the scaled execution time of the predecessor
            # NOTE: This assumes the predecessor runs on the *fastest possible* processor to calculate a baseline EST.
            # A more complex scheduler might consider the actual assigned processor of the predecessor.
            fastest_proc_speed = max(clock_speeds.values()) if clock_speeds else REFERENCE_SPEED_HZ
            pred_exec_time = processing_time.get(pred, 0) * (REFERENCE_SPEED_HZ / fastest_proc_speed)
            
            # Get communication cost
            comm_cost = communication_costs.get((pred, task), 0)
            
            # Calculate when the predecessor would be finished *including communication*
            predecessor_finish = earliest_start_time.get(pred, 0) + pred_exec_time + comm_cost
            
            if predecessor_finish > max_predecessor_finish_time:
                max_predecessor_finish_time = predecessor_finish
                
        earliest_start_time[task] = max_predecessor_finish_time
        
    return earliest_start_time

# Function to assign tasks to processors using list scheduling
def list_scheduling(graph, processing_time, communication_costs, processors ):
    processors_list, clock_speeds = processors # Unpack processors list and speeds
    
    # FIX: Pass clock_speeds to EST calculation
    earliest_start_time = calculate_earliest_start_time(graph, processing_time, communication_costs, clock_speeds)
    
    processor_available_time = {processor: 0 for processor in processors_list}
    schedule = []
    task_finish_times = {} # Store actual finish times for dependency checks
    
    # Iterate through each task in topological order
    for task in nx.topological_sort(graph):
        min_finish_time = float('inf')
        best_processor = None
        best_start_time = 0
        best_exec_time = 0

        # Find the processor that can finish the task the earliest
        for processor in processors_list:
            
            # 1. Calculate deterministic execution time for this processor
            clk_speed = clock_speeds.get(processor, REFERENCE_SPEED_HZ)
            exec_time = processing_time[task] * (REFERENCE_SPEED_HZ / clk_speed)
            
            # 2. Determine the earliest time the task can START on this processor
            # It must be after the processor is free AND after all dependencies are met.
            
            # Find the time the last required message arrives
            dependency_ready_time = 0
            for pred in graph.predecessors(task):
                pred_finish_time = task_finish_times.get(pred, 0)
                comm_cost = communication_costs.get((pred, task), 0)
                dependency_ready_time = max(dependency_ready_time, pred_finish_time + comm_cost)

            # The task can start after the processor is available AND dependencies are ready
            start_time = max(processor_available_time[processor], dependency_ready_time)
            
            finish_time = start_time + exec_time
            
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_processor = processor
                best_start_time = start_time
                best_exec_time = exec_time
        
        # Assign the task to the best processor found
        if best_processor is not None:
            processor_available_time[best_processor] = min_finish_time
            task_finish_times[task] = min_finish_time
            schedule.append((task, best_processor, best_start_time, min_finish_time))

    makespan = max(processor_available_time.values()) if processor_available_time else 0
    return schedule, makespan


# Function to Plot the schedule

def plot_list_schedule(final_schedule, processors, makespan):
    processors_list, _ = processors # Unpack only the list of processors for plotting
    plt.figure(figsize=(20, 10))
    color_map = {processor: f'C{processor}' for processor in processors_list}
    for idx, (task, processor, start_time, finish_time) in enumerate(final_schedule):
        plt.plot([start_time, finish_time], [idx, idx], linewidth=20, color=color_map[processor], solid_capstyle='butt')

    legend_handles = [Patch(facecolor=color_map[processor], label=f'Processor {processor}') for processor in processors_list]
    
    plt.xlabel('Time')
    plt.ylabel('Task')
    plt.title('Task Assignment to Processors')
    plt.yticks(range(len(final_schedule)), [f'Task {task}' for task, _, _, _ in final_schedule])
    plt.legend(handles=legend_handles, title='Processor', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim(-1, len(final_schedule))
    plt.grid(axis='y')
    # plt.show()  # Commented out to prevent blocking during batch validation