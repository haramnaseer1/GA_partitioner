import random   
from deap import base, creator, tools
from copy import deepcopy
from collections import defaultdict
from . import partitioning as pt
from . import reading_application_model as ra
from . import config as cfg
from . import List_Schedule as ls
from . import auxiliary_fun_GA as af
import os
import sys
import json
import logging
import networkx as nx
from itertools import islice
import numpy as np
import shutil
import dask
from dask.distributed import Client, LocalCluster, as_completed, wait, Queue as DaskQueue
import multiprocessing as mp
import re

# Set the Reference Speed for WCET scaling (1 GHz = 1e9 Hz)
REFERENCE_SPEED_HZ = 1e9

# --------------------------------------------------------------------------------------------------------------------------------------------
# Global Genetic Algorithm for the partitioning of the application model and scheduling of the tasks on the platform model
# The global GA is used to partition the application model into the three tier architecture and schedule the tasks on the platform model
# The global GA has two modes: normal mode and constrained mode
# --------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------
# Global GA Fitness Evaluation Function
# --------------------------------------------------------------------------------------------------------------------------------------------


# Local Genetic Algorithm
def local_ga(application_model, platform_model,clocking_speed,rsc_mapping,Processor_List, PopSzLGA, NGLGA,MTLGA,CXLGA,TSLGA,task_can_run_info,subGraphInfo):
    try:
        return _local_ga_impl(application_model, platform_model,clocking_speed,rsc_mapping,Processor_List, PopSzLGA, NGLGA,MTLGA,CXLGA,TSLGA,task_can_run_info,subGraphInfo)
    except Exception as e:
        # If local_ga fails, return a fallback schedule
        import traceback
        logging.error(f"local_ga failed for application {application_model}, platform {platform_model}: {e}")
        logging.error(traceback.format_exc())
        # Return empty schedule with high makespan to indicate failure
        return application_model, {}, 999999.0

def _local_ga_impl(application_model, platform_model,clocking_speed,rsc_mapping,Processor_List, PopSzLGA, NGLGA,MTLGA,CXLGA,TSLGA,task_can_run_info,subGraphInfo):
    dbg = False


    # Reading the application model
    with open(os.path.join(cfg.combined_SubGraph_dir_path, f"{application_model}.json")) as json_file_app:
        lam = json.load(json_file_app)
        #print("lam",lam)    
    
    # Reading the platform model
    with open(os.path.join(cfg.platform_dir_path, f"{platform_model}_Platform.json")) as json_file_plat:
        lpm = json.load(json_file_plat)
        
    
    # Reading the platform paths
    with open(os.path.join(cfg.path_info, "Paths.json")) as json_file_plat_path:
        platform_path = json.load(json_file_plat_path)

    
    # -------------- Defining the Variables -------------- #
    Num_Jobs  = len(lam['application']['jobs'])  # Number of Jobs
    msg = lam['application'].get('messages',[])  # Messages
    jbs = lam['application']['jobs']  # Jobs
    Num_Messages = len(msg)  # Number of Messages
    mkspan_indp_job = 0 # Initializing Makespan for Independent Job

    # -------------- Defining the Variables -------------- #

    # -------------- Defining the helper functions -------------- #

    # Function to read the application model
    def read_application_model(LAM):
        LAMx = LAM['application']
        return LAMx



    # Function to read the platform model
    def read_platform_model(LPM):
        LPMx = LPM['platform']
        return LPMx
    

    # Function to find communication cost
    def communication_costs(data):
        messages = data['messages']
        communication_costs = {}
        for message in messages:
            sender = message['sender']
            receiver = message['receiver']
            size = message['size']

            if sender not in communication_costs:
                communication_costs[sender] = {receiver: size}
            else:
                communication_costs[sender][receiver] = size

        return communication_costs

    # Function to find the processing time
    def processing_times(data):
        jobs = data['jobs']
        processing_times = [job['processing_times'] for job in jobs] 

        return processing_times

    # Function to find the message list 
    def Message_List(data):
        # Returns a list of dictionaries for each meassage attribute where the keys for each message will be
        # id,sender,reciever,size
        messages = data['messages']                              
        task_ids = [job['id'] for job in data['jobs']]           # creating a list of task_ids
        message_list = []                                        # Initializing a list
        for msg in messages:
            sender_id = task_ids.index(msg['sender'])
            receiver_id = task_ids.index(msg['receiver'])
            message_size = msg['size']
            message_id = msg['id']
            message_info = {
                'id': message_id,
                'sender': sender_id,
                'receiver': receiver_id,
                'size': message_size
            }
            message_list.append(message_info)
        return message_list
    

    # Function to change the ID of the tasks in the AM model to have a continuous ID.
    def change_task_id(AM):
        AMx1 = deepcopy(AM)
        jobs = AMx1['jobs']
        task_id = 0
        new_tk_id={}
        for job in jobs:
            
            new_tk_id[job['id']]=task_id
            job['id'] = task_id
            task_id += 1
        #print(new_tk_id)
        msg_id = 0
        for message in AMx1['messages']:
            message['sender'] = new_tk_id[message['sender']]
            message['receiver'] = new_tk_id[message['receiver']]
            message['id'] = msg_id
            msg_id += 1
        return AMx1, new_tk_id

    def compute_makespan(schedule):    # passing the Re-construction function result
        # Extract end times from the schedule
        end_times = [info[2] for info in schedule.values()]
        # The makespan is the maximum end time
        makespan = max(end_times)
        return makespan
    
    # Function to find the path mapping , Maps tasks to processors.
    # Updates the message list to include the assigned path index for communication between processors.
    
    def ComputeMappingsAndPaths(message_list, tasks, processors, message_orderings, path_indices):
        # Create a deep copy of the message list to avoid modifying the original
        message_list_copy = deepcopy(message_list)
        
        #print("message_orderings",message_orderings)
        # Create a dictionary that maps task IDs to processor IDs
        task_to_processor = {task: processors[i] for i, task in enumerate(tasks)}
        #print("task_to_processor",task_to_processor)
        # Container for the updated message list
        updated_message_list = []

        # Map message orderings to their respective path indices
        message_to_path_mapping = {message_orderings[i]: path_indices[i] for i in range(len(message_orderings))}
        #print("message_to_path_mapping",message_to_path_mapping)
        for message in message_list_copy:
            # Find the path index corresponding to the message ID
            path_index = message_to_path_mapping.get(message['id'], None)

            # If sender and receiver are on the same processor, set path_index to 0 (self-loop)
            if task_to_processor[message['sender']] == task_to_processor[message['receiver']]:
                path_index = 0

        

            # Create an updated message structure with the assigned path index
            updated_message = {
                'id': message['id'],
                'sender': task_to_processor[message['sender']],  # Map sender task to processor
                'receiver': task_to_processor[message['receiver']],  # Map receiver task to processor
                'size': message['size'],  # Retain message size
                'path_index': path_index  # Add path index for communication
            }

            # Append the updated message to the list
            updated_message_list.append(updated_message)

        return updated_message_list


    def find_suitable_paths(updated_message_list, merged_paths_dict):
        # Container for selected path IDs
        selected_path = []

        for message in updated_message_list:
            # Extract sender and receiver processor IDs
            sender = message['sender']
            receiver = message['receiver']
            # Convert path index to string for comparison
            path_index_str = str(message['path_index'])

            valid_path_ids = set()

            # Iterate through precomputed paths to find valid ones for this message
            for pid, details in merged_paths_dict.items():
                path = details['path']

                # Check if the path starts and ends at the correct sender and receiver
                if ((path[0] == sender and path[-1] == receiver) or
                    (path[0] == receiver and path[-1] == sender)) and pid.endswith(path_index_str):
                    valid_path_ids.add(pid)

            # Select a valid path ID if found; otherwise, use None
            path_id = next(iter(valid_path_ids), None)

            # Append the selected path ID to the list
            selected_path.append(path_id)

        # Replace None with '00' for paths that were not found
        selected_paths = ['00' if item is None else item for item in selected_path]

        return selected_paths
    
    # Function to evaluvate the fitness of the individual. The evaluvation is based on the makespan of the schedule.
    def optimise_schedule(processor_nodes_pltfm, processor_allocation, task_order, processing_time, message_list, selected_paths, platform_paths,
                        message_priority, communication_costs,clocking_speed):
        ###-------------------------------------###
        # selected_paths ==== message_path_index
        # platform_paths ==== all_path_indexes_with_costs

        # BUG FIX: Add epsilon to prevent exact overlap of tasks on same processor
        EPSILON = 1e-12  # Small time buffer between consecutive tasks

        # FIX: Validate and reassign invalid processors before scheduling
        processor_allocation = list(processor_allocation)  # Make mutable copy
        valid_processors = set(clocking_speed.keys())
        for idx, proc in enumerate(processor_allocation):
            if proc not in valid_processors:
                # Processor doesn't exist in current platform - reassign to first valid processor
                if valid_processors:
                    processor_allocation[idx] = list(valid_processors)[0]
                else:
                    # Use a default processor if none available
                    processor_allocation[idx] = 51

        message_list_copy = deepcopy(message_list)
        
        # BUG FIX: Preserve original message sizes before any modification
        original_message_sizes = {msg['id']: msg['size'] for msg in message_list_copy}
        
        num_processors = len(processor_nodes_pltfm)
        schedule = {}
        
        # Track task completion times by task_id (not index)
        task_completion_times = {}  # task_id -> end_time
        message_dict = defaultdict(list)  # Dictionary to store message details

        # Initialise s dict to map message ID to correspondinh priority
        message_priority_dict = {msg_id: priority for priority, msg_id in enumerate(message_priority)}  # Map message ID to priority

        # create a dict to map task to processor
        task_to_processor = {task_id: processor for task_id, processor in zip(task_order, processor_allocation)}

        # replacing the sender and receiver with the processor ID
        updated_msg_list = []
        for msg in message_list_copy:
            updated_msg = {
                'id': msg['id'],
                'sender': task_to_processor[msg['sender']],
                'receiver': task_to_processor[msg['receiver']],
                'size': msg['size']
            }
            updated_msg_list.append(updated_msg)

        # Create mapping from task_id to its index in task_order (needed early for message processing)
        task_id_to_index = {tid: idx for idx, tid in enumerate(task_order)}
        task_ids_in_partition = set(task_order)

        msg_to_path_mapping = []

        # Initialize a dictionary to track the usage of path_ids
        path_usage = {path_id: 0 for path_id in selected_paths}

        # print('message_path_index in reconstruction' ,message_path_index)

        # Loop through each message and find the corresponding path
        for i, message in enumerate(updated_msg_list):
            sender = message['sender']
            receiver = message['receiver']
            # Iterate through the message_path_index to find a matching path
            for path_id in selected_paths:
                #print(path_id)
                path_info = platform_paths[path_id]
                path = path_info['path']

                # Check if the sender and receiver match the path's start and end
                if (path[0] == sender and path[-1] == receiver) or (path[-1] == sender and path[0] == receiver):
                    # Check if this path has been used as many times as it appears in message_path_index
                    if path_usage[path_id] < selected_paths.count(path_id):
                        msg_to_path_mapping.append({'message_id': message['id'], 'path_id': path_id})
                        path_usage[path_id] += 1
                        break



        for idx, message in enumerate(message_list_copy):
            # Find the corresponding path_id from message_to_path_mapping
            message_mapping = next((m for m in msg_to_path_mapping if m['message_id'] == message['id']), None)
            # print("message_mapping",message_mapping)
            if message_mapping:
                path_id = message_mapping['path_id']
            else:
                path_id = '00'  # Set path_id to 00 if no mapping is found

            # Decode the path and cost using the path_id
            path = platform_paths[path_id]["path"]
            path_cost = platform_paths[path_id]["cost"]
            # print("path",path)
            # print("path_cost",path_cost)

            # BUG FIX: Separate original message size from communication delay
            sender_idx = task_id_to_index.get(message["sender"])
            receiver_idx = task_id_to_index.get(message["receiver"])
            
            original_size = original_message_sizes[message["id"]]
            comm_delay = original_size + path_cost  # Default: size + path cost
            
            # Check if both tasks are in this partition and on same processor
            if sender_idx is not None and receiver_idx is not None:
                sender_proc = processor_allocation[sender_idx]
                receiver_proc = processor_allocation[receiver_idx]
                
                if sender_proc == receiver_proc:
                    # Same processor - no communication delay (but keep original size)
                    comm_delay = 0
            
            # Store: (sender, comm_delay, priority, path_id, message_id, original_size)
            # comm_delay is used for timing, original_size is preserved for output
            message_dict[message["receiver"]].append((message["sender"], comm_delay, message_priority_dict[message["id"]], path_id, message["id"], original_size))


        # Sort each receiver's list in message_dict by message priority
        for receiver, messages in message_dict.items():
            message_dict[receiver] = sorted(messages, key=lambda x: x[2])

        # Track current time per processor to prevent overlaps
        current_time_per_processor = {k: 0 for k in processor_nodes_pltfm}
        
        # Track which tasks have been scheduled
        scheduled_tasks = set()
        pending_tasks = list(task_order)  # Work with task IDs directly
    
        # Process tasks respecting dependencies
        max_iterations = len(task_order) * len(task_order)
        iteration = 0
        
        while pending_tasks and iteration < max_iterations:
            iteration += 1
            tasks_scheduled_this_round = False
            
            # Try to schedule each pending task
            for task_id in list(pending_tasks):
                task_idx = task_id_to_index[task_id]
                processor = processor_allocation[task_idx]
                
                # Get clock speed with fallback
                clk_speed = clocking_speed.get(processor, 1.5e9)
                predecessors = message_dict[task_id]

                # Check if ALL predecessors are completed (both internal and external)
                # For external predecessors, we assume they're done and just need communication delay
                all_predecessors_ready = True
                for sender, _, _, _, _, _ in predecessors:  # FIX: 6-tuple (sender, comm_delay, priority, path_id, message_id, original_size)
                    if sender in task_ids_in_partition and sender not in scheduled_tasks:
                        all_predecessors_ready = False
                        break
                
                if all_predecessors_ready:
                    # All dependencies satisfied - schedule this task
                    
                    # Calculate earliest start time based on:
                    # 1. Predecessor completion times + communication delays
                    # 2. Processor availability (to prevent overlaps)
                    
                    earliest_start_from_deps = 0
                    for sender_id, comm_delay, _, _, _, _ in predecessors:
                        if sender_id in task_ids_in_partition:
                            # It's an internal predecessor, so its finish time is in the schedule
                            sender_finish_time = schedule[sender_id][2]
                            earliest_start_from_deps = max(earliest_start_from_deps, sender_finish_time + comm_delay)
                        else:
                            # It's an external predecessor, assume it finishes at time 0
                            # The start time is only dependent on the communication delay
                            earliest_start_from_deps = max(earliest_start_from_deps, comm_delay)
                    
                    # Calculate start time considering processor availability
                    start_time = max(current_time_per_processor[processor], earliest_start_from_deps)

                    # Calculate execution time
                    # CRITICAL FIX: Scale WCET (processing_time) by the ratio of Reference Speed (1 GHz) to Node Clock Speed (Hz)
                    if clk_speed > 0:
                        # Correct normalized execution time calculation
                        execution_time = processing_time[task_id] * (REFERENCE_SPEED_HZ / clk_speed)
                    else:
                        # Handle division by zero/zero speed case
                        execution_time = processing_time[task_id] # Assume 1:1 if speed is zero/unknown
                        
                    end_time = start_time + execution_time

                    # Record the schedule - BUG FIX: Store original message size, not communication delay
                    path_info = []
                    for sender, comm_delay, _, path_id, message_id, orig_size in predecessors:
                        # Always use original message size for dependencies (not communication delay)
                        path_info.append((sender, path_id, orig_size))
                    
                    schedule[task_id] = (processor, start_time, end_time, path_info)
                    
                    # Update tracking structures
                    task_completion_times[task_id] = end_time
                    current_time_per_processor[processor] = end_time  # Processor is busy until end_time
                    scheduled_tasks.add(task_id)
                    
                    # Remove from pending list
                    pending_tasks.remove(task_id)
                    tasks_scheduled_this_round = True
                    
                    # CRITICAL FIX: Break after scheduling ONE task to ensure processor times are updated
                    # This prevents multiple tasks from being scheduled on the same processor at the same time
                    break
            
            # If no tasks were scheduled this round and we still have pending tasks
            if not tasks_scheduled_this_round and pending_tasks:
                # Deadlock detected - schedule tasks in order ignoring dependencies
                # This shouldn't happen with valid DAG, but handle gracefully
                for task_id in list(pending_tasks):
                    task_idx = task_id_to_index[task_id]
                    processor = processor_allocation[task_idx]
                    
                    clk_speed = clocking_speed.get(processor, 1.5e9)
                    
                    # Start after processor is free (with epsilon buffer)
                    start_time = current_time_per_processor[processor]
                    if start_time > 0:
                        start_time += EPSILON
                        
                    if clk_speed > 0:
                        execution_time = processing_time[task_id] * (REFERENCE_SPEED_HZ / clk_speed)
                    else:
                        execution_time = processing_time[task_id]
                        
                    end_time = start_time + execution_time
                    
                    predecessors = message_dict[task_id]
                    path_info = [(sender, path_id, orig_size) for sender, _, _, path_id, _, orig_size in predecessors]
                    schedule[task_id] = (processor, start_time, end_time, path_info)
                    
                    task_completion_times[task_id] = end_time
                    current_time_per_processor[processor] = end_time
                    scheduled_tasks.add(task_id)
                    
                pending_tasks = []

        
        return schedule
    
        # Function to types of processors with processor id and the type of processor in the platform model
    def find_processor_type(data,rsc):
        # Create a reverse mapping of resource types for faster lookup
        valid_types = set(rsc.values())
        Nodes = data["nodes"]
        # Initialize an empty dictionary to hold the result
        processor_map = {}
        
        for node in Nodes:
            if not node.get("is_router", False) and "type_of_processor" in node:
                processor_type = node["type_of_processor"]
                if processor_type in valid_types:
                    # Add the node ID to the corresponding processor type
                    processor_map.setdefault(processor_type, []).append(node["id"])
        
        return processor_map
            
    


    # Function to find the processors on which the task can run on based on the "can run on" parameter in the application model.

    def find_processors_to_run(data, rsc, prcr_map):
        can_run_on = {} # Dictionary to store the processors on which the task can run on
        task_processor_mapping = {} # Dictionary to store the mapping of the task to the processors on which it can run on  
        for task in data["jobs"]:
            can_run_on[task["id"]] = task["can_run_on"]


        # Iterate through can_run_on keys and their values
        for task, processor_type_keys in can_run_on.items():
            # Initialize an empty list for processors that can run the task
            task_processor_mapping[task] = []
            
            for processor_type_key in processor_type_keys:
                # processor_type_key is the processor type ID (1-6)
                # Get the processor type name from rsc mapping
                processor_type_name = rsc.get(int(processor_type_key))
                
                # If the processor type exists in prcr_map, add its actual processor IDs to the task
                if processor_type_name in prcr_map:
                    task_processor_mapping[task].extend(prcr_map[processor_type_name])

        return can_run_on,task_processor_mapping

    # Function to update the schedule keys with the original task IDs
    def update_schedule_keys(final_schedule, updated_task_id):
        """
        Replace the keys of final_schedule with the corresponding keys from updated_task_id,
        and update the sender values in the nested lists of each tuple by comparing the mappings.

        Args:
            final_schedule (dict): The original schedule dictionary with new task IDs (0, 1, 2, ...)
            updated_task_id (dict): The mapping of original task IDs to new task IDs {original: new}

        Returns:
            dict: A new dictionary with original task IDs restored
        """
        # Create inverse mapping: new_id -> original_id
        inverse_mapping = {new_id: orig_id for orig_id, new_id in updated_task_id.items()}
        
        updated_schedule = {}

        for new_id_key, value in final_schedule.items():
            # Convert new ID back to original ID
            original_id = inverse_mapping.get(new_id_key, new_id_key)
            
            # Unpack the tuple value
            processor, start_time, end_time, dependencies = value
            
            # Update dependencies - convert sender IDs from new back to original
            # Dependencies store (sender_id, path_id, message_size)
            updated_dependencies = [
                (inverse_mapping.get(dep[0], dep[0]), dep[1], dep[2])
                for dep in dependencies
            ]
            
            # Create updated value tuple
            updated_value = (processor, start_time, end_time, updated_dependencies)
            
            # Update the schedule with the new key and updated value
            updated_schedule[original_id] = updated_value

        return updated_schedule
    
   
    
   
    # -------------------------------------------------------------------------------------------------------------------------------------#
    # Helper functions  and other conditions

    if Num_Jobs > Num_Messages and msg:
        # Determine jobs with messages
        jobs_with_message_ids = {msg1['sender'] for msg1 in msg}.union({msg1['receiver'] for msg1 in msg})

        jobs_with_message = [] # Jobs with messages
        independent_jobs = [] # Jobs without messages (independent jobs)

        for job1 in jbs:
            if job1['id'] in jobs_with_message_ids:
                jobs_with_message.append(job1)
            else:
                independent_jobs.append(job1)

        # Create the new JSON structure
        jobs_with_message_json = {'application': {'jobs': jobs_with_message, 'messages': msg}}
        independent_jobs_json = {'application': {'jobs': independent_jobs, 'messages': []}}

        mkspan_indp_job = sum([job['processing_times'] for job in independent_jobs]) # Add the processing time of independent jobs to find the makespan of independent jobs
        flg = 1 # A flag to indicate that the number of jobs is greater than the number of messages
    
    else:
        flg = 0 # A flag to indicate that the number of jobs is less than or equal to the number of messages
    
    
    # ------ Initilaization ------ #
    if flg ==1:
        LAM = read_application_model(jobs_with_message_json)
    else:
        LAM = read_application_model(lam)
    
    LPM = read_platform_model(lpm)
    #print("LAM",LAM)
    LAM_updated, updated_task_id = change_task_id(LAM) # Change the task ID to have a continuous ID and also updating the json data
    
    # DEBUG: Check message sizes BEFORE creating message_list
    if LAM_updated and 'messages' in LAM_updated:
        sizes = set([m['size'] for m in LAM_updated['messages']])
        print(f"DEBUG LAM_updated: {len(LAM_updated['messages'])} messages, unique sizes: {sorted(sizes)[:10]}")
    
    message_list = Message_List(LAM_updated) # Find the message list
    
    # DEBUG: Check message sizes AFTER creating message_list
    if message_list:
        sizes2 = set([m['size'] for m in message_list])
        print(f"DEBUG message_list: {len(message_list)} messages, unique sizes: {sorted(sizes2)[:10]}")
    communication_cost = communication_costs(LAM_updated) # Find the communication costs
    processing_time = processing_times(LAM_updated) # Find the processing time of the tasks   
    prcr_map = find_processor_type(LPM, rsc_mapping) # Find the processor type
    can_run_on,task_procr_map = find_processors_to_run(LAM_updated, rsc_mapping,prcr_map) # Find the processors on which the task in the DAG can run on and additionally map the task to the  which the tasks can run
    num_tk = len(processing_time) # Number of tasks
    
    # Build clocking_speed dict from loaded platform (LPM)
    local_clocking_speed = {}
    for node in LPM.get("nodes", []):
        if not node.get("is_router", False):  # Only processors, not routers
            processor_id = node.get("id")
            clk_speed_str = node.get("clocking_speed", "1.5 GHz")  # Default to 1.5 GHz
            # Convert string like "1.5 GHz" to Hz (float)
            clk_speed_hz = af.convert_clocking_speed_to_hz(clk_speed_str)
            local_clocking_speed[processor_id] = clk_speed_hz

    # -------------------------------------------------------------------------------------------------------------------------------------#
    if dbg:
        print("can_run_on",can_run_on)
        print("processing_time",processing_time)
        print("communication_cost",communication_cost)
        print("message_list",message_list)
        print("updated_task_id",updated_task_id)
        print("prcr_map",prcr_map)
        print("task_procr_map",task_procr_map)
    
    else:
        # print("can_run_on",can_run_on)
        # print("processing_time",processing_time)
        # print("communication_cost",communication_cost)
        # print("message_list",message_list)
        # print("updated_task_id",updated_task_id)
        # print("prcr_map",prcr_map)
        # print("task_procr_map",task_procr_map)
        cx = 0 # a random variable 
    
   
   
    # -------------------------------------------------------------------------------------------------------------------------------------#
    
    
    # ---------------------- Local Genetic Algorithm ---------------------- #
    def LGA(LAM_updated, LPM,message_list, communication_cost, processing_time, updated_task_id, clocking_speed, platform_path,can_run_on,task_procr_map,fileid,PopSzLGA, 
            NGLGA,MTLGA,CXLGA,TSLGA,task_can_run_info, rsc_mapping,subgraphinfo,application_model,platform_model,Processor_List):
        #print("Local GA Started")
        toolbox = base.Toolbox()    # Creating a toolbox for the global GA
        creator.create("fitnessMin", base.Fitness, weights=(-1.0,)) # Creating a fitness class for the global GA
        creator.create("individual", list, fitness=creator.fitnessMin) # Creating an individual class for the global GA


        # ----------------- Initialising the variables -----------------
        # Initialising the variables
        Num_tasks = len(processing_time)  # Number of tasks
        Num_Messages = len(message_list)  # Number of Messages

        # ------------------------------------------------ #


        
        # ---------------- Defining the chromosome for the Local GA ----------------

        # The chromosome consits of four parts: Task, processor allocation, path index, and message priority
        # The Task part consists of the Task of the application model that need to be scheduled
        # The processor allocation part consists of the allocation of the task to the processor in the assigned platform
        # The path index part consists of the path index between the processors
        # The chromosome is a list of lists
        # The first part of the chromosome is the Task, the second part is the processor allocation part, the third part is the path index part and the fourth 
        # part is the message priority part 

        # -------------- Defining the chromosomes -------------- #
        
        # Defining the Task part of the chromosome
        def init_tasks(Num_tasks):
            return random.sample(range(Num_tasks), Num_tasks)

        # Defining the processor allocation part of the chromosome
        def init_processor_allocation(task_procr_map, task_order):
            processor_alloc = [0] * len(task_order)
            for pos, task in enumerate(task_order):
                available_procs = task_procr_map.get(task, [])
                # Filter to only valid processors that exist in current platform
                valid_procs = [p for p in available_procs if p in local_clocking_speed]
                if valid_procs:
                    processor_alloc[pos] = random.choice(valid_procs)
                elif available_procs:
                    # No valid processors - use first available and log warning
                    processor_alloc[pos] = available_procs[0]
                    logging.warning(f"Task {task} has no valid processors in current platform, using {available_procs[0]}")
                else:
                    # Fallback - use first processor from platform
                    processor_alloc[pos] = Processor_List[0] if Processor_List else 51
                    logging.error(f"Task {task} has no processor mapping at all!")
            
            return processor_alloc
        
        # Wrapping the processor allocation part of the chromosome as it need information from the individual
        def init_processor_allocation_wrap(local_individual, task_procr_map, Num_tasks):
            tasklist = local_individual[0:Num_tasks]
            return init_processor_allocation(task_procr_map, tasklist)    
    
        # Defining the path index part of the chromosome
        def init_path_index(Num_Messages):
            return [random.choice([0, 1, 2, 3, 4]) for _ in range(Num_Messages)]
        
        # Defining the message priority part of the chromosome
        def init_message_priority(Num_Messages):
            return random.sample(range(Num_Messages), Num_Messages)
    
        # Defining the individual
        def init_local_individual():
            # Creating the individual
            local_individual = []
            # Adding the Task part of the chromosome
            local_individual.extend(toolbox.Task_Order())
            # Adding the processor allocation part of the chromosome
            local_individual.extend(toolbox.Processor_Allocation(local_individual))
            # Adding the path index part of the chromosome
            local_individual.extend(toolbox.Path_Index())
            # Adding the message priority part of the chromosome
            local_individual.extend(toolbox.Message_Priority())
            return local_individual
    
        # ------------------------------------------------ #

        # ---------------- Defining the fitness function for the Local GA ----------------
        def local_evaluation(individual, message_list, platform_path, processing_time, communication_cost,Processor_List):
            # Extracting the Task, processor allocation, path index, and message priority part of the chromosome
            task_order = individual[0:Num_tasks]
            processor_allocation = individual[Num_tasks:2*Num_tasks]
            message_path_index = individual[2*Num_tasks:2*Num_tasks + Num_Messages]
            message_priority = individual[2*Num_tasks + Num_Messages:]
            
            # Creating the updated message list
            Updated_Message_List = ComputeMappingsAndPaths(message_list, task_order, processor_allocation, message_priority,message_path_index )
            
            # Finding the suitable paths
            Selected_Paths = find_suitable_paths(Updated_Message_List, platform_path)
            
            # Optimising the schedule
            schedule = optimise_schedule(Processor_List, processor_allocation, task_order, processing_time, message_list, Selected_Paths, platform_path, message_priority, communication_cost, local_clocking_speed)
            
            # Evaluating the fitness of the individual
            makespan = compute_makespan(schedule)
            local_ga_fitness = makespan
            return local_ga_fitness,
    
        # --------------------------------------------------------------------------

        # ---------- Defining the mutation and crossover function ------------
    
        def l_crossover(l_ind1,l_ind2, Num_tasks):
            if dbg: 
                print("l_ind1 before",l_ind1)
                print("l_ind2 before",l_ind2)
            l_child1, l_child2 = toolbox.clone(l_ind1), toolbox.clone(l_ind2)
            tools.cxOnePoint(l_child1[:Num_tasks], l_child2[:Num_tasks])
            if dbg:
                print("l_child1 after",l_child1)
                print("l_child2 after",l_child2)
            return l_child1, l_child1 
        
        def l_mutation_task_order(l_ind, Num_tasks):
            task_odr = l_ind[:Num_tasks]
            if dbg: 
                print("task_odr before",task_odr)
            Indx = random.sample(range(Num_tasks), 2)
            task_odr[Indx[0]], task_odr[Indx[1]] = task_odr[Indx[1]], task_odr[Indx[0]]
            l_ind[0:Num_tasks] = task_odr
            if dbg: 
                print("task_odr after",task_odr)
            return l_ind,

        def l_mutation_processor_allocation(l_ind, Num_tasks,task_procr_map ):
            pro_alloc = l_ind[Num_tasks:2*Num_tasks]
            if dbg: 
                print("pro_alloc before",pro_alloc)
            tk_od = l_ind[:Num_tasks]
            for i in pro_alloc:
                if random.random() < 0.03:
                    for pos, task in enumerate(tk_od):
                        # Get the list of possible processors
                        pcr_list = task_procr_map.get(task, [])
                        # Filter to valid processors in current platform
                        valid_pcr_list = [p for p in pcr_list if p in local_clocking_speed]
                        
                        if not valid_pcr_list:
                            # No valid processors - use first from Processor_List
                            valid_pcr_list = Processor_List[:1] if Processor_List else [51]

                        # Only mutate if the list contains more than one processor
                        if len(valid_pcr_list) > 1:
                            pcr = random.choice(valid_pcr_list)

                            # Keep choosing a new processor until it's different
                            while pcr == pro_alloc[pos]:
                                pcr = random.choice(valid_pcr_list)
                        else:
                            # If there's only one option, just use it
                            pcr = valid_pcr_list[0]

                        # Assign the new processor
                        pro_alloc[pos] = pcr
            if dbg: 
                print("pro_alloc after",pro_alloc)
            l_ind[Num_tasks:Num_tasks + Num_tasks] = pro_alloc
            return l_ind,

        def l_mutation_path_index(l_ind, Num_tasks, Num_Messages):
            
            pth_index = l_ind[Num_tasks+Num_tasks:Num_tasks+Num_tasks + Num_Messages]
            if dbg: 
                print("path before",pth_index)
            for i in range(len(pth_index)):
                if random.random() < 0.03:
                    pth_index[i] = random.choice([0, 1, 2, 3, 4])
            l_ind[Num_tasks+Num_tasks:Num_tasks+Num_tasks + Num_Messages] = pth_index
            if dbg:  
                print("path after",pth_index)
            return l_ind,
        
        def l_mutation_message_priority(l_ind, Num_tasks, Num_Messages, message_list):
            msg_priority = l_ind[Num_tasks + Num_tasks+ Num_Messages:]
            if dbg: 
                print("msg_priority before",msg_priority)
            # FIX: Avoid mutation if msg_priority has less than 2 elements
            if len(msg_priority) >= 2:
                shf_msg = tools.mutShuffleIndexes(msg_priority, indpb=0.06)[0]
            else:
                shf_msg = msg_priority[:]
            if dbg: 
                print("shf_msg after",shf_msg)
            uq_id = list({msg['id'] for msg in message_list})
            random.shuffle(uq_id)

            seen = set()
            for i in range(len(shf_msg)):
                if shf_msg[i] in seen:
                    for ud in uq_id:
                        if ud not in seen:
                            shf_msg[i] = ud
                            seen.add(ud)
                            break
                    seen.add(shf_msg[i])
                else:
                    seen.add(shf_msg[i])
            if dbg: 
                print("msg_priority after",shf_msg)
            l_ind[Num_tasks + Num_tasks+ Num_Messages:] = shf_msg
            return l_ind,
        
    # --------------------------------------------------------------------------
        
    # ---------- Registering the functions with the toolbox ------------
    
        toolbox.register("Task_Order", init_tasks, Num_tasks=Num_tasks) # Registering the Task part of the chromosome
        toolbox.register("Processor_Allocation", init_processor_allocation_wrap, task_procr_map=task_procr_map, Num_tasks=Num_tasks) # Registering the processor allocation part of the chromosome
        toolbox.register("Path_Index", init_path_index, Num_Messages=Num_Messages) # Registering the path index part of the chromosome
        toolbox.register("Message_Priority", init_message_priority, Num_Messages=Num_Messages) # Registering the message priority part of the chromosome    

        toolbox.register("localindividual", tools.initIterate, creator.individual, init_local_individual) # Registering the individual
        toolbox.register("lg_population", tools.initRepeat, list, toolbox.localindividual) # Registering the population
        toolbox.register("lg_evaluate", local_evaluation, message_list=message_list, platform_path=platform_path, processing_time=processing_time, communication_cost=communication_cost, Processor_List=Processor_List) # Registering the evaluation function
        toolbox.register("lg_selection", tools.selTournament, tournsize=TSLGA) # Registering the selection function

        toolbox.register("lg_crossover", l_crossover, Num_tasks=Num_tasks)  # Registering the crossover function 
        toolbox.register("lg_mutation_task_order", l_mutation_task_order, Num_tasks=Num_tasks) # Registering the mutation function for the task order
        toolbox.register("lg_mutation_processor_allocation", l_mutation_processor_allocation, Num_tasks=Num_tasks,task_procr_map=task_procr_map) # Registering the mutation function for the processor allocation
        toolbox.register("lg_mutation_path_index", l_mutation_path_index, Num_tasks=Num_tasks, Num_Messages=Num_Messages) # Registering the mutation function for the path index
        toolbox.register("lg_mutation_message_priority", l_mutation_message_priority, Num_tasks=Num_tasks, Num_Messages=Num_Messages, message_list=message_list) # Registering the mutation function for the message priority
   
    # --------------------------------------------------------------------------
        local_population = toolbox.lg_population(n=PopSzLGA) # Creating the initial population

    # ------- Evaluating the fitness of the initial population ---------
        local_fitnesses = map(toolbox.lg_evaluate, local_population) # Evaluating the fitness of the initial population
        for lind, lfit in zip(local_population, local_fitnesses):
            lind.fitness.values = lfit
    
    # --------------------------------------------------------------------------

        # print(" Local Population",local_population)

        # ---------- Running the Local Genetic Algorithm ------------
        #print("Running the Local Genetic Algorithm")
        for lg_gen in range(NGLGA):
            #print("local_Generation",lg_gen)
            loffspring = toolbox.lg_selection(local_population, len(local_population)-1) # Selecting the offspring, and generating 99 offspring and the 100th is the best individual
            loffspring = list(map(toolbox.clone, loffspring)) # Cloning the offspring

            # Crossover
            for lgchild1, lgchild2 in zip(loffspring[::2], loffspring[1::2]):
                if random.random() < CXLGA:
                    toolbox.lg_crossover(lgchild1, lgchild2)
                    del lgchild1.fitness.values
                    del lgchild2.fitness.values
            
            # Mutation
            for lgmutant in loffspring:
                if random.random() < MTLGA:
                    toolbox.lg_mutation_task_order(lgmutant)
                    toolbox.lg_mutation_processor_allocation(lgmutant)
                    toolbox.lg_mutation_path_index(lgmutant)
                    toolbox.lg_mutation_message_priority(lgmutant)
                    del lgmutant.fitness.values
        
            # ---------- Evaluating the fitness of the offspring ---------
            # The fitness of the offspring is evaluated if the fitness is not evaluated
            new_local_individuals = [Lind for Lind in loffspring if not Lind.fitness.valid] # Extracting the individuals whose fitness is not evaluated
            new_local_fitnesses = map(toolbox.lg_evaluate, new_local_individuals)   # Evaluating the fitness of the offspring
            for lind, lfit in zip(new_local_individuals, new_local_fitnesses):  # Assigning the fitness to the individuals
                lind.fitness.values = lfit
            # ---------------------------------------------------------
            # ---------- Updating the population ---------
            best_local_individual = tools.selBest(local_population , 1)[0] # Selecting the best individual
            # print(f"Best Local Individual of application model {fileid} in {lg_gen}, {best_local_individual}")
            local_population[:] = loffspring + [best_local_individual] # Updating the population   
            if dbg:
                print("Local Population",local_population)    
        # --------------------------------------------------------- 

        final_best_local_individual = tools.selBest(local_population , 1)[0] # Selecting the best individual
        # print(f"Final Best Local Individual of application model {fileid} ",final_best_local_individual)
        
        # -------- Reconstruction of the final schedule ---------
        task_order_ = final_best_local_individual[0:Num_tasks]
        processor_allocation_ = final_best_local_individual[Num_tasks:2*Num_tasks]
        message_path_index_ = final_best_local_individual[2*Num_tasks:2*Num_tasks + Num_Messages]
        message_priority_ = final_best_local_individual[2*Num_tasks + Num_Messages:]
        
        # Creating the updated message list
        Updated_Message_List_ = ComputeMappingsAndPaths(message_list, task_order_, processor_allocation_,message_priority_, message_path_index_)
        
        # Finding the suitable paths
        Selected_Paths_ = find_suitable_paths(Updated_Message_List_, platform_path)
        
        # Optimising the schedule
        Finalschedule = optimise_schedule(Processor_List, processor_allocation_, task_order_, processing_time, message_list, Selected_Paths_, platform_path, message_priority_, communication_cost, local_clocking_speed)
        
        # Evaluating the fitness of the individual
        Finalmakespan = compute_makespan(Finalschedule)
        # print(f"Final Makespan of application model {fileid}",Finalmakespan)
        # print(f"Final Schedule application model {fileid}",Finalschedule)
        
        return Finalschedule,Finalmakespan


    # -------------------------------------------------------------------------------------------------------------------------------------#

    # -------- Defining conditions to take care of the following cases as result of partitioning  ---------

    # 1. If the partition is empty , this occurs as the Global GA runs it tries to optimise the partitions and as time progress some partitions may get empty.
    # 2. If the partition has only one task, which means there is no message in the partition
    # 3. If the partition has more than one task, but less than or equal to 2 task.
    # 4. If the partition has more than 2 tasks, then the Local GA is run to optimise the partition.

    # Case 1 
    fileid = application_model   
    if processing_time == [0]:
        FinalSchedule = {}
        FinalMakespan = 0  + mkspan_indp_job
        
    
    # Case 2
    elif num_tk == 1:
        tk_id = LAM["jobs"][0]["id"]
        pcr_Lt = task_procr_map[0] 
        if len(pcr_Lt) > 1:
            pa = random.choice(pcr_Lt)
        else:
            # If there's only one option, just use it
            pa = pcr_Lt[0]
        
        # BUG FIX: Divide processing time by clock speed
        clk_speed = local_clocking_speed.get(pa, 1.5e9)  # Get clock speed for selected processor
        
        if clk_speed > 0:
            execution_time = processing_time[0] * (REFERENCE_SPEED_HZ / clk_speed)
        else:
            execution_time = processing_time[0]

        
        FinalSchedule = {tk_id: (pa, 0, execution_time, [])}
        FinalMakespan = execution_time + mkspan_indp_job
        
    
    # Case 3
    elif 1 < num_tk <= 2:
        tk = list(range(num_tk))
        pa = [0] * len(tk)
        for pos, task in enumerate(tk):
            available_procs = task_procr_map.get(task, [])
            valid_procs = [p for p in available_procs if p in local_clocking_speed]
            if valid_procs:
                pa[pos] = random.choice(valid_procs)
            elif Processor_List:
                pa[pos] = Processor_List[0]
            else:
                pa[pos] = 51  # Fallback
        mg_num = len(message_list)
        mg_pidx = [random.choice([0, 1, 2, 3, 4]) for _ in range(mg_num)]
        mg_prty = list(random.sample(range(mg_num), mg_num))
        
        Uptd_Message_List_ = ComputeMappingsAndPaths(message_list, tk, pa,mg_prty, mg_pidx)
        Seld_Paths_ = find_suitable_paths(Uptd_Message_List_, platform_path)
        finalSchedule = optimise_schedule(Processor_List, pa, tk, processing_time, message_list, Seld_Paths_, platform_path, mg_prty, communication_cost, local_clocking_speed)
        
        # Evaluating the fitness of the individual
        FinalMakespan = compute_makespan(finalSchedule)
        FinalSchedule = update_schedule_keys(finalSchedule, updated_task_id) # Update the schedule keys with the original task IDs

        
    
    # Case 4
    else:
        finalSchedule,FinalMakespan = LGA(LAM_updated, LPM,message_list, communication_cost, processing_time, updated_task_id, local_clocking_speed, 
                                          platform_path,can_run_on,task_procr_map,fileid,PopSzLGA, NGLGA,MTLGA,CXLGA,TSLGA,task_can_run_info, rsc_mapping,subGraphInfo,application_model,platform_model,Processor_List)
        FinalSchedule = update_schedule_keys(finalSchedule, updated_task_id) # Update the schedule keys with the original task IDs
      
    # print("Final Schedule",FinalSchedule)
    # print("Final Makespan",FinalMakespan)   
    return application_model,FinalSchedule,FinalMakespan


# -------------------------------------------------------------------------------------------------------------------------------------#



# ---------------------Local GA END---------------------#



# -------------------------------------------------------------------------------------------------------------------------------------#







def global_ga_fitness_evaluation(AM,client,TaskNumInAppModel, partition_order, layer_order, inter_layer_path, subGraphInfo, selectedIndividual,partition_dependencies,genome_len,clocking_speed,rsc,pcr_list,inter_layer_message_priority,message_list_gl,
                                 vertex_edge_pairs,graph,task_can_run_info
                                 ):

    # ----------------- Initialising the variables -----------------
    # Initialising the variables
    
    s_ind = deepcopy(selectedIndividual) # Deep copying the selected individual
    p_dep = deepcopy(partition_dependencies) # Deep copying the partition dependencies  
    s_info = deepcopy(subGraphInfo) # Deep copying the subgraph info
    p_order = deepcopy(partition_order) # Deep copying the partition order
    l_order = deepcopy(layer_order) # Deep copying the layer order
    n_partitions = len(p_order) # Number of partitions

    popSzLGA = cfg.POPULATION_SIZE_LGA # Population size for the local GA
    nGLGA = cfg.NUMBER_OF_GENERATIONS_LGA # Number of generations for the local GA 
    mTLGA= cfg.MUTATION_PROBABILITY_LGA # Mutation probability for the local GA
    cXLGA = cfg.CROSSOVER_PROBABILITY_LGA # Crossover probability for the local GA
    tSLGA  = cfg.TOURNAMENT_SIZE_LGA # Tournament size for the local GA

    task_list = {} # Dictionary to store the tasks
    ready_partition = set(range(1, n_partitions+1))  # Set to store the ready partitions
    partition_ids = list(ready_partition) # List to store the partition ids
    pt_depp = {} # Creating a dictionary to store the partition dependencies
    pt_dep = {} # Creating a dictionary to store the partition dependencies after removing the cycles if created when new graphs are added  
    completed_partition = set() # Set to store the completed partitions
    Three_Tier_Schedule = {} # To store the three tier schedule
    results = {} # Dictionary to store the results after processing the files in the Dask cluster and local

    # -------------------------------------------------------------

    # ---------------- Functions ----------------
      
    # Converting the partition index of the selected individuals to the exact task and to the exact partition.
    for i,v in enumerate(s_ind):
        if v not in task_list:
            task_list[v] = [i] # If the task is not in the task list, add the task to the task list
        else:
            task_list[v].append(i) # If the task is in the task list, append the task to the task list
    
    fldr_to_save_SubGraph_GA = cfg.subGraph_dir_path # Folder to save the subgraph for the GA algorithm
    shutil.rmtree(fldr_to_save_SubGraph_GA, ignore_errors=True) # Removing the folder to save the subgraph for the GA algorithm
    os.makedirs(fldr_to_save_SubGraph_GA) # Creating the folder to save the subgraph for the GA algorithm

    for i in range(1, int(genome_len)+1):
        if i not in task_list: # If the task is not in the task list, add the task to the task list
            task_list[i] = []
    for k, pt in task_list.items(): # For each task in the task list
        af.convert_selInd_to_json(pt, AM,fldr_to_save_SubGraph_GA, k) # Convert the selected individual to json - use correct function name with capital I
    
    for partition in partition_ids:
        realation = []
        for dep_part, depend in p_dep.items():
            if partition in depend:
                realation.append(dep_part)
        pt_depp[partition] = realation
    pt_dep = af.find_and_remove_cycles(pt_depp) # Finding and removing the cycles 
    #print("Partition Dependencies crt: ", pt_dep) # Printing the partition dependencies

    # # --------------------------------------------------------------------------------------------
    
    # combining the subgraphs which have the same platform model assigned in the GA chromosomes
    
    layer_partition_dict = {}
    # print("Layer Order: ", layer_order)
    # print("Partition Order: ", partition_order) 
    for l in set(layer_order):  # Iterate through unique values in layer_order
        indices = [i for i, val in enumerate(layer_order) if val == l]  # Find all indices of the layer
        ps = [partition_order[i] for i in indices]  # Get corresponding partition values
        layer_partition_dict[l] = ps  # Store in dictionary
    
    # print("Layer Partition Dict: ", layer_partition_dict)

    Combine_SubGph_Layer_Dict = {} # Dictionary to store the combined subgraph for each layer (The key is the combined subgraph and the value is the layer)
    shutil.rmtree(cfg.combined_SubGraph_dir_path, ignore_errors=True) # Removing the folder to save the subgraph for the GA algorithm
    os.makedirs(cfg.combined_SubGraph_dir_path) # Creating the folder to save the subgraph for the GA algorithm
    for index, (lr, pats) in enumerate(layer_partition_dict.items()):
        af.combine_subgraphs(pats, index, cfg.combined_SubGraph_dir_path, fldr_to_save_SubGraph_GA )
        Combine_SubGph_Layer_Dict[index+1] = lr
    
    # print("Layer_Combine_SubGph_Dict: ", Combine_SubGph_Layer_Dict)
        
            

    # # --------------------------------------------------------------------------------------------
  

    files = [os.path.join(cfg.combined_SubGraph_dir_path, f) for f in os.listdir(cfg.combined_SubGraph_dir_path) if f.endswith(".json")]
    if not files:
        raise ValueError("No JSON files found in the directory.") 

    # # Use Dask workers count instead of CPUs
    workers = len(client.scheduler_info()['workers'])
    if not workers:
        workers = 32
    num_processes = min(workers, len(files))
    results = {}
    
    

    # List to store the futures (tasks) submitted to workers
    futures = []
    # Detect platform from application filename (same logic as in main.py and auxiliary_fun_GA.py)
    app_name = cfg.file_name  # e.g., "T2_var_003.json"
    match = re.match(r'[Tt](\d+)_', app_name)
    if match:
        platform_model_str = match.group(1)  # e.g., "2" for T2_var_003
    else:
        # Fallback for non-standard names (T20.json, TNC100.json, etc.)
        platform_model_str = "5"  # Use Platform 5 as default
        if not os.path.exists(os.path.join(cfg.platform_dir_path, f"{platform_model_str}_Platform.json")):
            platform_model_str = "3"  # If Platform 5 doesn't exist, use Platform 3
    
    print(f"Using Platform {platform_model_str} for application {app_name}")
    print("----------------- Running in Parallel Mode -----------------")
    # Submit each file to the Dask cluster one by one
    for file in files:
        # Extract the numeric ID from the file name (assuming the digits in the name represent the ID)
        file_id = int("".join(filter(str.isdigit, file)))  # Extract numeric ID from filename
        
        # Use the detected platform for all partitions
        assigned_layer = platform_model_str
        # Submit the task to a Dask worker: la.local_ga is the function to process the file
        #future = client.submit(local_ga,file_id ,assigned_layer,clocking_speed,rsc,pcr_list,popSzLGA, nGLGA,mTLGA,cXLGA,tSLGA,task_can_run_info)  # Submit task to Dask worker with the file name, platform assigned 
        # print("****************************************************************************************************")
        # print("file_id: ", file_id) 
        # print("assigned_layer: ", assigned_layer)
        future = client.submit(local_ga,file_id ,assigned_layer,clocking_speed,rsc,pcr_list,popSzLGA, nGLGA,mTLGA,cXLGA,tSLGA,task_can_run_info,subGraphInfo)  # Submit task to Dask worker with the file name, platform assigned 
        #future = local_ga(file_id ,assigned_layer,clocking_speed,rsc,pcr_list,popSzLGA, nGLGA,mTLGA,cXLGA,tSLGA,task_can_run_info,subGraphInfo)  # Submit task to Dask worker with the file name, platform assigned 
        
        # print("****************************************************************************************************")
        # Append the future to the list of futures
        futures.append(future)
    print("----------------- Tasks submitted -----------------")

    # Process tasks as they complete
    for completed_future in as_completed(futures):
        try:
            res = completed_future.result()
            #print("Result: ", res)
            k, tdict, mk = res
            results[k] = (tdict, mk)
        except Exception as e:
            print(f"Task failed: {e}")
            # Continue processing other partitions even if one fails
    cfg.global_ga_logger.info("------------------------------------------------------------------------------------")
    print(f"All files processed. Results: {results}")


    # # --------------------------------------------------------------------------------------------
    TaskListApplModel = set(range(0, TaskNumInAppModel))  # Task list of the application model 
    TaskListApplModelList = list(TaskListApplModel) # Task list of the application model as a list
    gl_processor_allocation = [] # Global processor allocation list

    # Populate the processor_allocation list
    for task in TaskListApplModelList:
        allocated_processor = None
        for key, (task_dict, _) in results.items():
        #for key, va in results.items():
            #tid, task_dict,*oth = va    
            # print("key: ", key)
            # print("task_dict: ", task_dict)
            if task in task_dict:
                allocated_processor = task_dict[task][0]
                break
        gl_processor_allocation.append(allocated_processor)
    #print("gl_processor_allocation: ", gl_processor_allocation)
    gl_tk_list = af.create__task_list(selectedIndividual)
    Depnd = af.find_dependencies(graph)
    partition_dependencies_task,pd = af.find_partition_dependencies(gl_tk_list,Depnd) 
    task_depencies_tuple = [ tpl for outer_key in partition_dependencies_task for inner_key in partition_dependencies_task[outer_key] for tpl in partition_dependencies_task[outer_key][inner_key]]    
     # Finding the Mapping and the path of the inter layer communication
    # Reading the platform paths
    with open(cfg.path_info + "/Paths.json") as json_file_plat_path:
    #with open(  "../Path_Information/Paths.json") as json_file_plat_path:
        Platform_Path = json.load(json_file_plat_path)
    
    MapandPath = af.ComputeMappingsAndPathsGlobal(message_list_gl, TaskListApplModelList, gl_processor_allocation, inter_layer_message_priority,inter_layer_path,partition_order,layer_order,selectedIndividual,task_depencies_tuple,subGraphInfo)
    Selected_Paths = af.find_suitable_pathsGlobal(MapandPath, Platform_Path)
    schedule_differences = {}

    for partition, (tasks, _) in results.items():
    # for partition, (task_dict,_) in results.items():
    #     td,tasks,*ot = vl
        dep_differences = {}
        for task, (processor, start_time, end_time, dependencies) in tasks.items():
            for sender_task, pathid, pathindex in dependencies:
                # Get the end time of the sender task
                sender_end_time = tasks[sender_task][2]
                # Calculate the time difference
                time_difference = start_time - sender_end_time
                # Store the result
                dep_differences[(sender_task,task)] = time_difference
        if dep_differences:
            schedule_differences[partition] = dep_differences

    updated_schedule = af.update_schedule_with_dependencies(results, task_depencies_tuple, Selected_Paths,schedule_differences)
    UpdatedSch = af.remove_duplicates_from_schedule(updated_schedule)
    UpdatedSchedule, partition_maxtime_Pair = af.update_schedule_times(UpdatedSch)
    
    cfg.global_ga_logger.info(f"updated_schedule : {UpdatedSchedule}")
    gl_mkspan = af.globalMakespan(partition_maxtime_Pair)
    cfg.global_ga_logger.info(f"Global Makespan: {gl_mkspan}")  # Log the global makespan
    cfg.global_ga_logger.info("------------------------------------------------------------------------------------")  
    return gl_mkspan


# --------------------------------------------------------------------------------------------------------------------------------------------



if cfg.operating_mode == "constrain":
 

    def global_ga_constrained(AM, client,G, constrained_task_copy,selected_individual, subGraphInfo,list_schedule_makespan,adjacency_matrix,
                             genome_len,clocking_speed,rsc_mapping, processor_list,message_list_gl,vertex_edge_pairs,graph, task_can_run_info, partition_dependencies):
    

    
 
        print("Global GA Started")
        toolbox = base.Toolbox()    # Creating a toolbox for the global GA
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Creating a fitness class for the global GA
        creator.create("Individual", list, fitness=creator.FitnessMin) # Creating an individual class for the global GA

        # --------- Defining variables to store values ------------

        # Variables to store the best individual and best fitness value
        best_glindividual = []
        best_glfitness_value = [] 

        # Dictionary to store the best individual and best fitness value in a generation 
        best_glindividual_dict = {}
        best_glfitness_value_dict = {}

        # variables for storing additional statistics
        mean_fitness_per_gen = []  # Mean fitness per generation
        std_fitness_per_gen = []   # Standard deviation of fitness per generation

    
        # -----------------------------------------------------------

        # initialising the global GA parameters
        # print("Selected Individual: ", selectedIndividual)      
        num_partitions = len(subGraphInfo) # Number of partitions
        # print("Number of Partitions: ", num_partitions)
        num_platforms = cfg.num_layers # Number of platforms 
        TaskNumInAppModel = len(AM['application']['jobs']) # Number of tasks in the application model

        # -----------------------------------------------------------

        # ---------------- Defining the chromosome for the global GA ----------------

        # The chromosome consits of the partitioning and platform allocation
        # The partitioning part consists of the partitioning of the application model 
        # The platform allocation part consists of the allocation of the partitions to the platforms (three tier architecture)
        # The chromosome is a list of lists
        # The first part of the chromosome is the partitioning part and the second part is the platform allocation part


            # ----- Defining the partitioning part of the chromosome -----

        def init_partitioning(): 
            # It creates a sequence of integers from 1 to num_partitions (inclusive) and shuffles them using random.sample to ensure no duplicates.
            return random.sample(range(1, num_partitions + 1), num_partitions)

            # ----- Defining the platform allocation part of the chromosome -----

        def init_platform_allocation(num_layers, constrained_task, Task_can_run_info, partition_order, subGraphInfo):
            #print("Partition Order: ", partition_order)
            #print("SubGraphInfo: ", subGraphInfo)
            #print("Constrained Task: ", constrained_task)
            #print("Fixed Subgraph Map: ", fixed_subgraph_map)


            # Initialize the layers list with zeros.
            layers = [0] * len(partition_order)

            # Iterate over the partitions to determine layer assignments.
            for pos, sub in enumerate(partition_order):
                # Check if the partition exists in the subGraphInfo dictionary.
                if sub in subGraphInfo.keys():
                    tks = subGraphInfo[sub]  # Retrieve tasks associated with the subgraph.

                    if not tks:
                        # If the subgraph is empty, assign a random layer to avoid errors.
                        layers[pos] = random.choice(range(1, num_layers + 1))
                        # print("Empty lAYER : ", layers)
                    else:
                        # Iterate through tasks in the subgraph.
                        for tk in tks:
                            if tk in constrained_task:
                                # If the task is removed, assign a layer from the edge (1,9).
                                layers[pos] = random.choice(range(1,10))
                                # print("Constrained Task: ", layers)
                                break
                            else:
                                # If the task is not removed, assign a random layer to the partition.
                                for x, can_run in Task_can_run_info.items():
                                    if any(val in [5, 6] for val in can_run):  # Check if any element in can_run is 5 or 6
                                        layers[pos] = random.choice([10,11,12,13,14])
                                        break
                                        # print("Task can run: ", layers)
                                    else:
                                        layers[pos] = random.choice(range(1, num_layers + 1))
                                    # print("Task can run: ", layers)
                                
                                # print(tk)
                                # print("Random Layer: ", layers) 
                                break

            # Return the final layer assignments.
            return layers
        def init_layer_wrapper(individual, num_partitions, num_layers, constrained_task, Task_can_run_info, subGraphInfo):
            """
            Wrapper function to initialize layer assignments for a given individual.

            Args:
                individual (list): A genetic encoding where the first `num_partitions` elements represent the partition order.
                num_partitions (int): Total number of partitions to consider.
                num_layers (int): Total number of available layers.
                constrained_task (list): Tasks that have been removed or are unavailable.
                fixed_subgraph_map (dict): Mapping of tasks to the layers where they can execute.
                subGraphInfo (dict): Mapping of partitions to their respective task lists.

            Returns:
                list: A list of layer assignments for each partition in the individual's order.
            """
            # Extract the partition order from the individual.
            partition_order = individual[:num_partitions]


            # Call init_layers to generate layer assignments based on the partition order and constraints.
            return init_platform_allocation(num_layers, constrained_task, Task_can_run_info, partition_order, subGraphInfo)
        

        def init_inter_layer_path_index(num_partitions):
            # Generates an array of random inter layer path indices for each partition.
            # `[random.choice[] for _ in range(num_partitions)]` creates an array of size `num_partitions`,
            # with each element being randomly chosen from the range 1 to 4.
            return [random.choice([0, 1, 2, 3, 4]) for _ in range(num_partitions)]
              
        
        # Defining the inter layer message priority part of the chromosome
        def init_inter_layer_message_priority(num_partitions):
            return random.sample(range(num_partitions), num_partitions)

        # --------------------------------------------------------------------------

        # ---------- Defining the individual ------------

        def init_gl_individual():
            # Initialising the individual
            global_individual = []
            # Initialising the partitioning part of the chromosome
            global_individual.extend(toolbox.partitions())
            # Initialising the platform allocation part of the chromosome
            global_individual.extend(toolbox.platform_allocation(global_individual))
            # Initialising the inter layer path index part of the chromosome
            global_individual.extend(toolbox.inter_layer_path_index())
            # Initialising the inter layer message priority part of the chromosome
            global_individual.extend(toolbox.inter_layer_message_priority())
            return global_individual


        # --------------------------------------------------------------------------
        
        # ---------- Defining the evaluation function ------------
        def global_evaluation(individual,subGraphInfo,selectedIndividual,genome_len):
            partition_length = len(individual) // 4
            layer_length = len(individual) // 4
            partition_order = individual[:partition_length]
            layer_order = individual[layer_length:layer_length+partition_length]
            inter_layer_path = individual[partition_length+layer_length:partition_length+layer_length+partition_length]
            inter_layer_message_priority = individual[partition_length+layer_length+partition_length:]

            
            # Finding the Mapping and the path of the inter layer communication


            
            # Calculate the fitness value of the individual.
            # The fitness value is calculated based on the partition order and layer order. using the below function.
            
            # Variable to check if the fitness value is improved
          
            
            global_ga_fitness = global_ga_fitness_evaluation(AM,client,TaskNumInAppModel,partition_order, layer_order, inter_layer_path, subGraphInfo, 
                                                             selectedIndividual,partition_dependencies,genome_len,clocking_speed,rsc_mapping, 
                                                             processor_list,inter_layer_message_priority,message_list_gl,vertex_edge_pairs,graph,task_can_run_info)    
            
            
            
            if global_ga_fitness > list_schedule_makespan:
                #print("SubGraphInfo: ", subGraphInfo)   
                subGraphInfo_ = af.update_individual(adjacency_matrix,subGraphInfo,constrained_task_copy,task_can_run_info)
                #print("Updated SubGraphInfo: ", subGraphInfo_)
                selectedIndividual = af.convert_partition_to_task_list(subGraphInfo_)
                subGraphInfo = subGraphInfo_
                #print("Updated Selected Individual: ", selectedIndividual)

            return global_ga_fitness,
        

        # --------------------------------------------------------------------------

        # ---------- Defining the mutation and crossover function ------------

        def crossover(ind1, ind2):
            part_length = len(ind1) // 4
            ind_clone1, ind_clone2 = toolbox.clone(ind1), toolbox.clone(ind2)
             # Perform crossover only on the remaining part of the genome (after part_length)
            ind_clone1[part_length:], ind_clone2[part_length:] = (
                ind_clone2[part_length:],  # Swap the second half of ind1 with ind2
                ind_clone1[part_length:],  # Swap the second half of ind2 with ind1
            )

            if cfg.DEBUG_MODE:
                print("Crossover")
                print("ind1: ", ind1)
                print("ind2: ", ind2)
                print("ind_clone1: ", ind_clone1)
                print("ind_clone2: ", ind_clone2)
            return ind_clone1, ind_clone2

        def mutate_partition(individual):
            gen_len = len(individual) // 4
            partition =individual[:gen_len]
            if cfg.DEBUG_MODE: 
                print("Partition before Mutation: ", partition)
            # FIX: Avoid mutation if partition has less than 2 elements
            if len(partition) >= 2:
                mutated_partition = tools.mutShuffleIndexes(partition, indpb=cfg.indprb)[0]
            else:
                mutated_partition = partition[:]
            #print("mutated_partition: ", mutated_partition)
            individual[0:gen_len] = mutated_partition
            if cfg.DEBUG_MODE:
                print("Mutate Partition")
                print("individual: ", individual)   
            return individual,

        def mutate_layer(individual):
            gen_len = len(individual) // 4
            layer = individual[gen_len:gen_len+gen_len]
            if cfg.DEBUG_MODE:
                print("Layer before Mutation: ", layer)
            
            # FIX: Check if we have enough elements to swap
            if gen_len >= 2:
                indx = random.sample(range(gen_len), 2)
                layer[indx[0]], layer[indx[1]] = layer[indx[1]], layer[indx[0]]
            
            #print("layer: ", layer)
            individual[gen_len:gen_len+gen_len] = layer
            if cfg.DEBUG_MODE:
                print("Mutate Layer")
                print("individual: ", individual)
            return individual,

        def mutate_inter_layer_path(individual):
            gen_len = len(individual) // 4
            interpath = individual[gen_len+gen_len:gen_len+gen_len+gen_len]
            if cfg.DEBUG_MODE: 
                print("interpath before Mutation: ", interpath)
            # FIX: Avoid mutation if interpath has less than 2 elements
            if len(interpath) >= 2:
                updated_interpath = tools.mutShuffleIndexes(interpath, indpb=cfg.indprb)[0]
            else:
                updated_interpath = interpath[:]
            #print("updated_interpath: ", updated_interpath)
            individual[gen_len+gen_len:gen_len+gen_len+gen_len] = updated_interpath
            if cfg.DEBUG_MODE:
                print("Mutate Inter Layer Path")
                print("individual: ", individual)   
            return individual,

        def mutate_inter_layer_message_priority(individual):
            gen_len = len(individual) // 4
            inter_message_priority = individual[gen_len+gen_len+gen_len:]
            if cfg.DEBUG_MODE: 
                print("inter_message_priority before Mutation: ", inter_message_priority)
            # FIX: Avoid mutation if inter_message_priority has less than 2 elements
            if len(inter_message_priority) >= 2:
                Shf_Msg = tools.mutShuffleIndexes(inter_message_priority, indpb=cfg.indprb)[0]
            else:
                Shf_Msg = inter_message_priority[:]
            #print("updated_inter_message_priority: ", Shf_Msg)
            Unq_id = list({msgl['id'] for msgl in message_list_gl})
            random.shuffle(Unq_id)
            Seen = set()
            for i in range(len(Shf_Msg)):
                if Shf_Msg[i] in Seen:
                    for ud in Unq_id:
                        if ud not in Seen:
                            Shf_Msg[i] = ud
                            Seen.add(ud)
                            break
                    Seen.add(Shf_Msg[i])
                else:
                    Seen.add(Shf_Msg[i])
            
            individual[gen_len+gen_len+gen_len:] = Shf_Msg
            if cfg.DEBUG_MODE:
                print("Mutate Inter Layer Message Priority")
                print("individual: ", individual)   
            return individual,


        # --------------------------------------------------------------------------
        
        # ---------- Registering the functions with the toolbox ------------

        toolbox.register("partitions", init_partitioning) # Registering the function to initialise the partitioning part of the chromosome
        toolbox.register("platform_allocation", init_layer_wrapper, num_partitions=num_partitions, num_layers=num_platforms, constrained_task=constrained_task_copy, Task_can_run_info=task_can_run_info, subGraphInfo=subGraphInfo) # Registering the function to initialise the platform allocation part of the chromosome 
        toolbox.register("inter_layer_path_index", init_inter_layer_path_index, num_partitions =num_partitions) # Registering the function to initialise the inter layer path index part of the chromosome
        toolbox.register("inter_layer_message_priority", init_inter_layer_message_priority, num_partitions =num_partitions) # Registering the function to initialise the inter layer message priority part of the chromosome
    
    
        toolbox.register("gl_individual", tools.initIterate, creator.Individual, init_gl_individual) # Registering the function to initialise the global individual
        toolbox.register("gl_population", tools.initRepeat, list, toolbox.gl_individual) # Registering the function to initialise the global population

        toolbox.register("gl_evaluate", global_evaluation, subGraphInfo=subGraphInfo,selectedIndividual=selected_individual,genome_len=genome_len) # Registering the evaluation function
        toolbox.register("select", tools.selTournament, tournsize=3) # Registering the selection function
        
        toolbox.register("mate", crossover) # Registering the crossover function
        toolbox.register("mutate_partition", mutate_partition) # Registering the mutation function for the partitioning part of the chromosome
        toolbox.register("mutate_layer", mutate_layer, ) # Registering the mutation function for the platform allocation part of the chromosome
        toolbox.register("mutate_inter_layer_path", mutate_inter_layer_path) # Registering the mutation function for the inter layer path index part of the chromosome
        toolbox.register("mutate_inter_layer_message_priority", mutate_inter_layer_message_priority) # Registering the mutation function for the inter layer message priority part of the chromosome
        

      

        print("ind", init_gl_individual())
        
        # --------------------------------------------------------------------------

        global_population = toolbox.gl_population(n=cfg.POPULATION_SIZE_GGA) # Initialising the global population

        # ---------- Evaluating the fitness of the initial population ------------
        global_fitnesses = map(toolbox.gl_evaluate, global_population) # Evaluating the fitness of the initial population
        glft_initial = deepcopy(global_fitnesses)
        global_fitnessesses = list(glft_initial)
        for gind, gfit in zip(global_population, global_fitnesses):
            gind.fitness.values = gfit    # Assigning the fitness values to the individuals

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # Compute initial population fitness statistics
        initial_max_fitness = max(fit[0] for fit in global_fitnessesses)  # Maximum fitness
        initial_min_fitness = min(fit[0] for fit in global_fitnessesses)  # Minimum fitness
        initial_avg_fitness = np.mean([fit[0] for fit in global_fitnessesses])  # Average fitness

        print("Initial Population: Max Fitness: ", initial_max_fitness)
        print("Initial Population: Min Fitness: ", initial_min_fitness)
        print("Initial Population: Avg Fitness: ", initial_avg_fitness)
        # Log the initial population fitness stats
        cfg.global_ga_logger.info(f"Initial Population: Max Fitness: {initial_max_fitness:.3f}, Min Fitness: {initial_min_fitness:.3f}, Avg Fitness: {initial_avg_fitness:.3f}")







        # ---------- Printings ---------------
        if cfg.DEBUG_MODE:
            print("num_partitions: ", num_partitions)
            print("num_platforms: ", num_platforms)
            po = init_partitioning()
            print("Partition Order: ", po)
            # print("Platform Allocation: ", init_platform_allocation(num_platforms, constrained_task, fixed_subgraph_map, po, subGraphInfo))
            print("global_population: ", global_population)
        # --------------------------------------------------------------------------
        else:
            # print("Global Population", global_population)
            # print("num_partitions: ", num_partitions)
            # print("num_platforms: ", num_platforms)
            # po = init_partitioning()
            # print("Partition Order: ", po)
            # print("Platform Allocation: ", init_platform_allocation(num_platforms, constrained_task, fixed_subgraph_map, po, subGraphInfo))
            # print("global_population: ", global_population)
            cx=0 # a random variable



        # ---------- Running the global GA ------------
        print("Running Global GA")
        for gl_generation in range (1, cfg.NUMBER_OF_GENERATIONS_GCA+1):
            print("Global Generation: ", gl_generation)
            gloffspring = toolbox.select(global_population, len(global_population))   # Selecting the offspring
            gloffspring = list(map(toolbox.clone, gloffspring))    # Cloning the offspring

            # ---------- Crossover ------------
            for glchild1, glchild2 in zip(gloffspring[::2], gloffspring[1::2]):
                if random.random() < cfg.CROSSOVER_PROBABILITY_GGA:
                    #print("glchild1: ", glchild1)   
                    #print("glchild2: ", glchild2)   
                    toolbox.mate(glchild1,glchild2)

                    del glchild1.fitness.values
                    del glchild2.fitness.values
            
            # ---------- Mutation ------------
            for glmutant in gloffspring:
                if cfg.DEBUG_MODE:
                    print("individual selected for mutation: ", glmutant)
                if random.random() < cfg.MUTATION_PROBABILITY_GGA:
                    toolbox.mutate_partition(glmutant)
                    toolbox.mutate_layer(glmutant)
                    toolbox.mutate_inter_layer_path(glmutant)
                    toolbox.mutate_inter_layer_message_priority(glmutant)
                    del glmutant.fitness.values

        
            # ---------- Evaluating the fitness of the offspring ------------
            # The individuals whose fitness values are not calculated are evaluated again

            new_gl_individuals = [glind for glind in gloffspring if not glind.fitness.valid] # Getting the individuals whose fitness values are not calculated
            global_fitnesses = map(toolbox.gl_evaluate, new_gl_individuals) # Evaluating the fitness of the individuals
            for glind, gfit in zip(new_gl_individuals, global_fitnesses): # Assigning the fitness values to the individuals
                glind.fitness.values = gfit
            # --------------------------------------------------------------------------

            # ---------- Replacing the population with the offspring ------------

            best_indd_gl = tools.selBest(global_population, 1)[0] # Getting the best individual in the population
            global_population[:] = gloffspring # Replacing the population with the offspring

            # ---------- Updating the best individual and best fitness value ------------
            best_glfitness_value.append(best_indd_gl.fitness.values[0]) # Updating the best fitness value
            best_glindividual.append(best_indd_gl)
            best_glindividual_dict[gl_generation] = best_indd_gl
            best_glfitness_value_dict[gl_generation] = best_indd_gl.fitness.values[0]

            # ---------- Additional statistics ------------
            mean_fitness_per_gen.append(np.mean([gind.fitness.values[0] for gind in global_population])) # Mean fitness per generation
            std_fitness_per_gen.append(np.std([gind.fitness.values[0] for gind in global_population])) # Standard deviation of fitness per generation

            # ---------- Printings ---------------
            if cfg.DEBUG_MODE:
                print(f"Generation {gl_generation}: Mean Fitness: {mean_fitness_per_gen[-1]:.3f}, Std: {std_fitness_per_gen[-1]:.3f}")
                print("Best Individual of Generation: ", best_glindividual_dict)
                print("Fitness of Best Individual: ", best_glfitness_value_dict)
            else:
                print(f"Generation {gl_generation}: Mean Fitness: {mean_fitness_per_gen[-1]:.3f}, Std: {std_fitness_per_gen[-1]:.3f}")
                print("Best Individual of Generation: ", best_glindividual_dict)
                print("Fitness of Best Individual: ", best_glfitness_value_dict)

            # --------------------------------------------------------------------------
        print("Global GA Completed")
        return best_glindividual_dict, best_glfitness_value_dict, mean_fitness_per_gen, std_fitness_per_gen



# --------------------------------------------------------------------------
# Auxiliary functions
# --------------------------------------------------------------------------

# =====================================================================
# TIER MAPPING AND LATENCY CONSTANTS (FROM Guide.pdf §10.7)
# =====================================================================

def task_can_run_on_which_processor(application_data, platform_data):
    """
    Determines which processors each task can run on based on eligibility.
    """
    task_can_run_info = {}
    # FIX: Access the 'tasks' list directly from the application data root
    for task in application_data.get('tasks', []):
        task_id = task['id']
        eligible_processors = []
        for processor in platform_data['platform']['processors']:
            if processor['type'] in task['eligible_p_types']:
                eligible_processors.append(processor['id'])
        task_can_run_info[task_id] = eligible_processors
    return task_can_run_info

# =====================================================================
#                        SCHEDULING LOGIC
# =====================================================================