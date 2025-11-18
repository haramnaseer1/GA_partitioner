# # Importing the libraries

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import re
from collections import defaultdict
from decimal import Decimal
from random import randint, sample
from functools import partial
from deap import base, creator, tools, algorithms
from copy import deepcopy
import shutil
import subprocess
import plotly.graph_objects as go
import time
import logging 
import sys
import dask
from dask.distributed import Client, LocalCluster, Queue as DaskQueue
import multiprocessing as mp
import datetime
from . import partitioning as pt
from . import reading_application_model as ra
from . import config as cfg
from . import List_Schedule as ls
from . import auxiliary_fun_GA as af
from . import global_GA as ga
import argparse
import os, time, logging




   


# processing the application model.
def main_file(client):

    AM = ra.read_application_model() # Reading the application model
    processing_time = ra.find_processing_time(AM) # Finding the processing time of the tasks in the Application Model
    communication_cost = ra.find_communication_cost(AM) # Finding the communication time of the tasks in the Application Model
    sender_receiver_pair = ra.find_sender_receiver_pairs(AM) # Finding the sender and receiver pairs in the Application Model where key is the center and value is the receivers
    vertex_edge_pairs = ra.find_sender_receiver_pairs_tuple(sender_receiver_pair)
    message_list_gl = ra.find_message_list(AM)
    adjacency_matrix = ra.find_adjacency_matrix(vertex_edge_pairs)
    graph = ra.plot_graph(vertex_edge_pairs)
    G = deepcopy(graph)


    if cfg.operating_mode == "constrain":
        start_nodes,end_nodes = af.finding_the_StartEnd_node(graph) # Finding the start and end nodes of the graph

    if cfg.DEBUG_MODE:
        print("Processing Time: ", processing_time)
        print("Communication Cost: ", communication_cost)
        print("Sender Receiver Pair: ", sender_receiver_pair)
        print("Vertex Edge Pairs: ", vertex_edge_pairs)
        print("Message List: ", message_list_gl)
        print("Adjacency Matrix: ", adjacency_matrix)
        print("Graph: ", graph) 
        if cfg.operating_mode == "constrain":
            print("Start Nodes: ", start_nodes)
            print("End Nodes: ", end_nodes)
    else:
        print("Graph: ", graph)
        ass=0 # random variable
       
        

    # Generating the subgraphs for the application model and population
    rsc_mapping = af.resource_mapping() # Resource Mapping
    node_job_mapping = af.map_job_ids_to_nodes(AM)   # Mapping the job ids to the nodes in which it can run on with the help of the json data of application model

    constrained_task = []  # Initialize empty for nonconstrain mode
    constrained_task_copy = []
    
    if cfg.operating_mode == "constrain":
        print(f"[DEBUG] Graph before constraint removal: {graph.number_of_nodes()} nodes")
        print(f"[DEBUG] Graph nodes: {list(graph.nodes())}")
        modified_graph, constrained_task = af.remove_constrained_task_from_graph(graph, node_job_mapping)
        print(f"[DEBUG] Removed constrained tasks: {constrained_task}")
        print(f"[DEBUG] Graph after constraint removal: {modified_graph.number_of_nodes()} nodes")
        print(f"[DEBUG] Remaining nodes: {list(modified_graph.nodes())}") 
        constrained_task_copy = constrained_task.copy()
        all_sets_of_subgraphs = pt.run_multiple_times_constrainmode(modified_graph, cfg.num_runs,constrained_task) # Running the multiple times to get the subgraphs
        
    else:
        all_sets_of_subgraphs = pt.run_multiple_times(graph, cfg.num_runs)


    AllSubGraph = pt.extract_nodes_from_subgraphs(all_sets_of_subgraphs) # Extracting the nodes to create the dict of all subgraphs from the number of runs
    total_vertex = af.total_vertex(AM) # Total number of vertices in the graph
    folder_name = cfg.subgraph_dir_path # Path for Subgraph Json Files
    shutil.rmtree(folder_name, ignore_errors=True)
    os.makedirs(folder_name)
    for run_key, subgraphs in all_sets_of_subgraphs.items():
        current_part = 1
        for subgraph in subgraphs:
            pt.extract_info_of_subgraph(run_key, subgraph, AM, current_part,folder_name) # Extracting the information of the subgraph
            current_part += 1
    subpopulation = pt.create_population(AllSubGraph, total_vertex) # Creating the population

    if cfg.DEBUG_MODE:
        print("All Sets of Subgraphs: ", all_sets_of_subgraphs)
        print("All SubGraph: ", AllSubGraph)
        print("Subpopulation: ", subpopulation)
    else:
        if cfg.operating_mode == "constrain":
            # print("constrained_task: ", constrained_task)
            aa=0 # random variable   
        # print("Subpopulation: ", subpopulation)
        aa =0 # random variable

    # Running the List Schedule to get the deadline of the tasks
    ls_pltf = ls.read_platform_model() # Finding the deadline of the tasks in the Application Model
    ls_communication_cost = ls.communication_costs_task(AM) # Finding the communication time of the tasks in the Application Model  
    ls_processing_time = ls.finding_ProcessTime(AM) # Finding the processing time of the tasks in the Application Model
    ls_processor = ls.find_processors(ls_pltf) # Finding the processor of the tasks in the Application Model  
    list_schedule, list_schedule_makespan = ls.list_scheduling(G,ls_processing_time, ls_communication_cost, ls_processor) # Running the List Schedule

    if cfg.DEBUG_MODE:
        ls.plot_list_schedule(list_schedule,ls_processor,list_schedule_makespan)
        print("List Schedule: ", list_schedule)
        print("List Schedule Makespan: ", list_schedule_makespan)
    else:
        #print("List Schedule: ", list_schedule)
        print("List Schedule Makespan: ", list_schedule_makespan)


    # Running the auxiliary functions for the Genetic Algorithm

    genome_length, selected_individual,selindx= af.select_individual(subpopulation) # Selecting the individual from the population
    subg,subGraphInfo = af.extract_subgraph_info(all_sets_of_subgraphs,selindx) # Extracting the subgraph information from the selected individual for the GA , this has the key pair and also selecting graph corresponding to the selected individual
    globaltask_list = af.create__task_list(selected_individual) # Creating the task list from the individual selected from the population
    graph = ra.plot_graph(vertex_edge_pairs)
    dependencies = af.find_dependencies(graph)  # Finding the dependencies between the tasks in the orginal Application model
    partition_dependencies_with_task, partition_dependencies = af.find_partition_dependencies(globaltask_list, dependencies) # Finding the partition dependencies between the tasks in the task list
    sorted_key_counts = af.key_count_partition(partition_dependencies_with_task) # Sorting the key counts of the partition
    missing_partition = af.find_missing_partition(sorted_key_counts, globaltask_list) # Finding the missing partition

    if cfg.DEBUG_MODE:
        print("Genome Length: ", genome_length)
        print("Selected Individual: ", selected_individual)
        print("Global Task List: ", globaltask_list)
        print("Dependencies: ", dependencies)
        print("Partition Dependencies with Task: ", partition_dependencies_with_task)
        print("Partition Dependencies: ", partition_dependencies)
        print("Sorted Key Counts: ", sorted_key_counts)
        print("Missing Partition: ", missing_partition)
        print("Subgraph Info: ", subGraphInfo)
        print("Subgraph: ", subg)
        
    else:
        # print("selected_individual: ", selected_individual) 
        print("Genome Length: ", genome_length) 
        # print("partition_dependencies_with_task: ", partition_dependencies_with_task)
        # print("partition_dependencies: ", partition_dependencies)
        # print("subgraphinfo: ", subGraphInfo)   

    # #-----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Processing the platform model

    # The functions defined (generate_processor_paths) in this block runs only if the there is a change in the platform model json file. 
    # This is because computing the path for three -tier platform model is computationally expensive and time consuming. 

    # Find the last modified time of the platform model json file
    # Detect platform number from application filename
    # (re and os already imported at module level)
    match = re.match(r'[Tt](\d+)_', cfg.file_name)  # Match T#_var format only
    if match:
        platform_num = match.group(1)
        # FIX: Check if specific path file exists
        paths_json_file = cfg.path_info + f"/Paths_{platform_num}.json"
    else:
        # Fallback for non-standard names 
        paths_json_file = cfg.path_info + "/Paths.json"
    
    
    # Check if path info file exists 
    if os.path.exists(paths_json_file):
        current_modified_time = os.path.getmtime(paths_json_file)
    else:
        # If the specific path file doesn't exist, treat as modified (0)
        current_modified_time = 0
    # print("current_modified_time: ", current_modified_time)

    pt_graph, processor_list, pcocessor_found_only_edge, clk_speed_GHz = af.load_graph_from_json() # Reading the platform model
                # pt_graph: Platform model graph
                # processor_list: List of the processors in the platform model
                # pcocessor_found_only_edge: List of the processors found in the edge layers of the platform model
                # clk_speed_GHz: Clock speed of the processors in the platform model in GHz

    clocking_speed = {processor_id: af.convert_clocking_speed_to_hz(speed) for processor_id, speed in clk_speed_GHz.items()} # Finding the clock speed of the processors in the platform model in Hz
    
    task_can_run_info = ga.task_can_run_on_which_processor(AM, client)
    #print("task_can_run_info",task_can_run_info)
    
    fixed_subgraph_map = af.fixed_subgraph_map2platform(constrained_task,task_can_run_info,rsc_mapping,pcocessor_found_only_edge)
    #print("fixed_subgraph_map: ", fixed_subgraph_map)

    if current_modified_time == cfg.last_known_time:
        # FIX: Load the correct platform-specific paths file
        with open(paths_json_file) as f:
            merged_path_dict = json.load(f)
    else:
        # Generate paths and save to the determined platform-specific paths_json_file
        path_dict, merged_path_dict = af.generate_processor_paths(processor_list, pt_graph,cfg.num_path) # Generating the processor paths
        # NOTE: generate_processor_paths saves to the correct platform file name internally

    if cfg.DEBUG_MODE:
        print("Processor List: ", processor_list)
        print("Processor Found Only on Edge: ", pcocessor_found_only_edge)
        print("Clock Speed GHz: ", clk_speed_GHz)
        print("Clocking Speed in Hz: ", clocking_speed)
        # print("Path Dict: ", path_dict)
        # print("Merged Path Dict: ", merged_path_dict)
        print("last_known_time: ", cfg.last_known_time)
        print("current_modified_time: ", current_modified_time)
    else:
        print("-- Path Generated --  ")

    #Assuming you're using the operating mode to select GA
    if cfg.operating_mode == "constrain":
        ga.global_ga_constrained(AM, client,G, constrained_task_copy,selected_individual, subGraphInfo,list_schedule_makespan,adjacency_matrix,
                                 genome_length,clocking_speed,rsc_mapping, processor_list,message_list_gl,vertex_edge_pairs,graph, task_can_run_info, partition_dependencies)
    else:
        ga.global_ga(AM, client,graph, selected_individual, subGraphInfo,list_schedule_makespan,adjacency_matrix, partition_dependencies, genome_length,clocking_speed,processor_list,rsc_mapping,message_list_gl,vertex_edge_pairs)  # Running the Global GA in Unconstrained Mode
        # after ga.global_ga(...) or ga.global_ga_constrained(...)
        # --- make last_schedule.json right after GA ---
        
       


    # #-----------------------------------------------------------------------------------------------------------------------------------------------------#

# #---------------------------------------------------- Main -------------------------------------------------




if __name__ == "__main__":
    # file_path = "/Logs/global_ga.log"
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    # else:
    #     print("The file does not exist")
    
    # Initialize Dask client (local cluster setup)
    
    # Handle command line arguments for application file
    if len(sys.argv) >= 3:
        app_file_arg = sys.argv[2]
        # Extract just the filename from the path
        if '/' in app_file_arg or '\\' in app_file_arg:
            cfg.file_name = os.path.basename(app_file_arg)
        else:
            cfg.file_name = app_file_arg
        print(f"Application model Selected: {cfg.file_name}")

    if cfg.cluster == "omni":
        # Start Dask on Omni cluster
        print("Starting Dask on Omni cluster")
        scheduler_host= sys.argv[1]
        if not scheduler_host:
            print("Scheduler host not provided")
            sys.exit(1)
        print(f"Scheduler_host: {scheduler_host}")
        client = Client(f"tcp://{scheduler_host}:32306")  # Connect to Omni scheduler
    else:
        # Start a LocalCluster
        print("Starting Dask on local cluster")
        num_cores = max(1,15)
        print(f"Number of cores: {num_cores}")
        cluster = LocalCluster(n_workers=num_cores, threads_per_worker=1, processes=False, dashboard_address=':9000')
        print("Dask dashboard at http://localhost:9000")
        client = Client(cluster)

    
    main_file(client)  # Run the main file    

    # Close Dask client
    client.close()  # Close Dask client
    print("Dask client closed")# # Importing the libraries