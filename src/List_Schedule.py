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

# *************************** Functions ***************************

# Reading the input file (Json file)


# Function to read the Application Model from the JSON file 
# def application_model(json_data):
#     AM = json_data['application']
#     return AM

# Function to read the Platform Model from the JSON file 
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



# Function to find the processors in the platform model
def find_processors(PM):
    processors = []
    for i in range(len(PM['nodes'])):
        if PM['nodes'][i]['is_router'] == False:
            processors.append(PM['nodes'][i]['id'])
    return processors





# ********************* List Scheduling Algorithm *********************


# Function to calculate earliest start time for each task
def calculate_earliest_start_time(graph, processing_time, communication_costs ):
    earliest_start_time = {}
    for task in nx.topological_sort(graph):
        max_comm_time = 0
        for sender, receiver in graph.in_edges(task):
            max_comm_time = max(max_comm_time, earliest_start_time[sender] + communication_costs[(sender, receiver)])
        earliest_start_time[task] = max_comm_time
    return earliest_start_time

# Function to assign tasks to processors using list scheduling
def list_scheduling(graph, processing_time, communication_costs, processors ):
    # Calculate earliest start time for each task
    earliest_start_time = calculate_earliest_start_time(graph, processing_time, communication_costs)
    #print("Earlist Start Time",earliest_start_time)
    # Initialize the finish time for each processor
    finish_time = {processor: 0 for processor in processors}
    
    
    # Initialize the schedule list
    schedule = []
    mk = []
    # Iterate through each task in topological order
    for task in nx.topological_sort(graph):
        min_finish_time =    float('inf')
        selected_processor = None
        #print("Min Finish Time",min_finish_time)
         
        
        # Find the processor with the minimum finish time
        for processor in processors:

                #start_time = max(earliest_start_time[task], finish_time[processor])
                #finish_time_task = start_time + processing_time[task]  
                #print(start_time, finish_time_task, processor,task)
                #if finish_time_task < min_finish_time:
                #    min_finish_time = finish_time_task
                #    selected_processor = processor
                    
                #    processor_assigned.append(processor)
            
            #elif (len(processor_assigned) == len(processors)):
            #    processor_assigned = []

            start_time = max(earliest_start_time[task], finish_time[processor])
            finish_time_task = start_time + processing_time[task] + 0.5*random.randint(5,10) # adding a random number to accound the cost of processor communication
            #print(start_time, finish_time_task, processor,task)
            if finish_time_task < min_finish_time:
                min_finish_time = finish_time_task
                selected_processor = processor
            
        
        # Update finish time for the selected processor
        finish_time[selected_processor] = min_finish_time
        #print("Finish Time",finish_time)
        mk.append(finish_time_task)
        
        # Append task and processor to the schedule
        schedule.append((task, selected_processor, start_time, finish_time_task))
        max_time = max(mk)
    return schedule, max_time



# Function to Plot the schedule

def plot_list_schedule(final_schedule, processors, makespan):
    plt.figure(figsize=(20, 10))
    color_map = {processor: f'C{processor}' for processor in processors}
    for idx, (task, processor, start_time, finish_time) in enumerate(final_schedule):
        plt.plot([start_time, finish_time], [idx, idx], linewidth=20, color=color_map[processor], solid_capstyle='butt')

    legend_handles = [Patch(facecolor=color_map[processor], label=f'Processor {processor}') for processor in processors]
    
    plt.xlabel('Time')
    plt.ylabel('Task')
    plt.title('Task Assignment to Processors')
    plt.yticks(range(len(final_schedule)), [f'Task {task}' for task, _, _, _ in final_schedule])
    plt.legend(handles=legend_handles, title='Processor', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim(-1, len(final_schedule))
    plt.grid(axis='y')
    # plt.show()  # Commented out to prevent blocking during batch validation






    
