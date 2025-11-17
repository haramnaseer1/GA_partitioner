from . import config as cfg 
import networkx as nx
import random
import numpy as np
import json
import os
import sys
from itertools import islice
import re
from copy import deepcopy


# Selecting the individual from the population created above
# The individual consists of the unique partitioning number.

def select_individual(subPopulation): 
    k = random.choice(list(subPopulation.keys())) # Selecting the individual at random from the population
    selected_graph_part = subPopulation[k] # Selecting the individual at random from the population

    # Find the max value presnt in the individual to find the length of the  genome.
    # The length of the genome is equal to the number of partitios in the individual
    genome_length = max(selected_graph_part)
    return genome_length, selected_graph_part,k

# Function to create the  task list from the individual selected from the population.
def create__task_list(selected_ind):
    # Create a dictionary to store the task that needs to be run on edge fog and cloud
    edge_fog_cloud_task = {}
    for index, value in enumerate(selected_ind):
        if value not in edge_fog_cloud_task:
            edge_fog_cloud_task[value] = [index]
        else:
            edge_fog_cloud_task[value].append(index)
    
    return edge_fog_cloud_task

# Function to find the dependencies between the task in the orginal Application model.
# The dependencies are found by finding the edges in the graph and then finding the dependencies between the tasks.
def find_dependencies(gh):
    # Find the edges in the graph
    edges = gh.edges()
    #print("Edges:", edges)
    # Find the dependencies between the tasks
    dependencies = []
    for edge in edges:
        dependencies.append((edge[0], edge[1]))
    return dependencies


# Function to find the partition dependencies between the tasks in the task list
def find_partition_dependencies(Tasksdep, dependencies):
    partition_dependencies_with_task = {}

    for edge in dependencies:
        source_task, target_task = edge
        source_partition = None
        target_partition = None

        # Find partitions for source and target tasks
        for partition, tasks in Tasksdep.items():
            if source_task in tasks:
                source_partition = partition
            if target_task in tasks:
                target_partition = partition

        # If tasks belong to different partitions, add dependency
        if source_partition != target_partition:
            if source_partition not in partition_dependencies_with_task:
                partition_dependencies_with_task[source_partition] = {}
            if target_partition not in partition_dependencies_with_task[source_partition]:
                partition_dependencies_with_task[source_partition][target_partition] = []

            partition_dependencies_with_task[source_partition][target_partition].append((source_task, target_task))
   

    partition_dependencies = {key: list(value.keys()) for key, value in partition_dependencies_with_task.items()}
   
    return partition_dependencies_with_task,partition_dependencies


# Function to find the key of the partition that has more dependencies
def key_count_partition(part_dep):
    key_counts = {}

    for sub_dict in part_dep.values():
        for key in sub_dict:
            key_counts[key] = key_counts.get(key, 0) + 1

    
    sorted_key_counts = dict(sorted(key_counts.items(), key=lambda item: item[1]))
   
    return sorted_key_counts

# Function to find the partitions mising in the sorted key counts that is in the key_count partition function
def find_missing_partition(sorted_key_counts, edgefogcloud_task):
    
    for i in edgefogcloud_task:
        if i not in sorted_key_counts:
            sorted_key_counts[i] = 0
            
    
    sorted_missing_partition = dict(sorted(sorted_key_counts.items(), key=lambda item: item[1]))
   
    return sorted_missing_partition



# Funcrion to find the start and end time of the tasks in the partition based on the dependencies 
def initialize_start_end_time(partition, makespan):
    repeating_values_dict = {}
    for key, value in partition.items():
        if value in repeating_values_dict:
            repeating_values_dict[value].append(key)
        else:
            repeating_values_dict[value] = [key]

    sorted_repeating_values_dict = dict(sorted(repeating_values_dict.items()))
    
    start_time = 0
    start_time_dict = {}
    end_time_dict = {}  

    for key, values in sorted_repeating_values_dict.items():
        for value in values:
            start_time_dict[value] = start_time
        start_time += 50
   
    

    for key, value in start_time_dict.items():
        
        end_time_dict[key] = round(value + 20*makespan/100 + 20*value/100)
   

    return start_time_dict, end_time_dict


# Function to conver the partitions into json file for later process.
def convert_selind_to_json(partition, jsondata, folder_name, k):

    extracted_info = {"application": {"jobs":[] , "messages": []}}
    filename = f"{folder_name}/individual_part_{k}.json"
    if not partition:
        extracted_info["application"]["jobs"].append({
                    "id": 0,
                    "wcet_fullspeed": 0,
                    "mcet": 0,
                    "deadline": 0,
                    "can_run_on": 0,
                    "processing_times": 0
                })
        extracted_info["application"]["messages"].append({
                            "id": 0,
                            "sender": 0,
                            "receiver":0,
                            "size": 0,
                            "period": 0
                        })
    else:
        for nd in partition:
            for job in jsondata["jobs"]:
                if nd == job["id"]:
                    
                    extracted_info["application"]["jobs"].append({
                        "id": job["id"],
                        "wcet_fullspeed": job["wcet_fullspeed"],
                        "mcet": job["mcet"],
                        "processing_times": job["processing_times"],
                        "deadline": job["deadline"],
                        "can_run_on": job["can_run_on"]
                        
                    })
        
        for message in jsondata["messages"]:
            if message["sender"] in partition and message["receiver"] in partition:
                extracted_info["application"]["messages"].append({
                            "id": message["id"],
                            "sender": message["sender"],
                            "receiver": message["receiver"],
                            "size": message["size"],
                            "timetriggered": message["timetriggered"],
                            "period": message["period"]
                        })

    
    
    with open(filename, "w") as outfile:
        json.dump(extracted_info, outfile, indent=2)



# Function to combine the subgraphs that run on same platform
def combine_subgraphs(pats,lr, com_fldr, sub_fldr):

    com_fl_dt = {"application": {"jobs":[] , "messages": []}}
    com_fl = f"{com_fldr}/{lr+1}.json"
   
    for i in pats:
        with open(sub_fldr+ f"/{i}.json") as subfl:
            sub_fl_dt = json.load(subfl)
            com_fl_dt["application"]["jobs"].extend(sub_fl_dt["application"]["jobs"])
            com_fl_dt["application"]["messages"].extend(sub_fl_dt["application"]["messages"])
        
        
    with open(com_fl, "w") as comfl:
        json.dump(com_fl_dt, comfl, indent=2)



# Function to remove the task from the graph and create subgraphs as it can be run only on certain nodes
# (nodes that is only found in edge layers) (Raspberry Pi, Micro-Controller) before partitioning.

def remove_constrained_task_from_graph(graph, node_job_mapping):
    # Create a deep copy of the original graph
    task_ids = []
    for k in node_job_mapping.keys():
        if k in [3,4]:  # Check for specific keys (Raspberry Pi, Micro-Controller)
            for task_id in node_job_mapping[k]:
                # Add task only if it's not already in the list
                if task_id not in task_ids:
                    task_ids.append(task_id)
    # Remove the specified task nodes from the modified graph

    graph.remove_nodes_from(task_ids)

    return graph, task_ids



def fixed_subgraph_map2platform(removed_tasks, task_can_run_info, rsc_mapping, unique_edge_pcr):
    fixedsubgrap2plt = {}

    for task_id in task_can_run_info.keys():
        if task_id in removed_tasks:
            for rsc in task_can_run_info[task_id]:
                if rsc in range(1, 7):  # Simplified the resource check
                    rsc_type = rsc_mapping[rsc]
                
                    # Check if the resource type exists in unique_edge_pcr
                    if rsc_type in unique_edge_pcr:
                        if task_id not in fixedsubgrap2plt:
                            # Initialize the list of platforms for this task
                            fixedsubgrap2plt[task_id] = list(unique_edge_pcr[rsc_type])
                        else:
                            # Extend the list of platforms for this task
                            fixedsubgrap2plt[task_id].extend(unique_edge_pcr[rsc_type])

            fixedsubgrap2plt[task_id] = list(set(fixedsubgrap2plt[task_id]))


    return fixedsubgrap2plt


# Function defining the type of resources available in the platform model
def resource_mapping():
    rsc_mapping = {
    1: 'General purpose CPU',
    2: 'FPGA',
    3: 'Raspberry Pi 5',
    4: 'Microcontroller',
    5: 'High Performance CPU',
    6: 'Graphical Processing Unit'
}

    return rsc_mapping


def map_job_ids_to_nodes(jsondata):
    jobs = jsondata["application"]["jobs"]
    # Initialize an empty dictionary to store the mapping
    node_to_job_ids = {}

    # Iterate over each job
    for job in jobs:
        # Get the list of nodes where this job can run
        can_run_on_nodes = job.get("can_run_on", [])

        # For each node, add this job's ID to the corresponding entry in the dictionary
        for node in can_run_on_nodes:
            if node not in node_to_job_ids:
                node_to_job_ids[node] = []
            node_to_job_ids[node].append(job["id"])

    return node_to_job_ids


def total_vertex(jsondata):
    # Total number of vertices in the child graph
    total_vertex_jsondata = len(jsondata["application"]["jobs"])
    return total_vertex_jsondata


def finding_the_StartEnd_node(graph):
    # Find starting nodes (nodes with no incoming edges)
    starting_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]

    # Find end nodes (nodes with no outgoing edges)
    end_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

    return starting_nodes, end_nodes


def extract_subgraph_info(all_sets_of_subgraphs,k):
    # # Initialize a dictionary to store the nodes in each subgraph
    # subgraph_dict = {}
    # subgh_dict = {} # dict to store the nodes in each subgraph intermediate


    # # Iterate over the list of population and extract the nodes and assign them to the correct subgraph
    # for i, index in enumerate(pop):
    #     for j, indx in enumerate(pop):
    #         if index == indx:
    #             if index not in subgh_dict :
    #                 subgh_dict[index] = [j]
    #             elif j not in subgh_dict[index]:
    #                 subgh_dict[index].append(j)

    # sub_index = 1

    # # Modifies the subgraphs such that the task on raspberry pi, microcontroller are split to different subgraphs so that can be used in the gA for constraints

    # for i, k in subgh_dict.items():

    #     if i == 0:
    #         for j in k:
    #             subgraph_dict[sub_index] = [j]
    #             sub_index += 1
    #     else:
    #         subgraph_dict[sub_index] = k
    #         sub_index += 1

    # return subgraph_dict

    subgraph = {}
    
    # Iterate over all subgraph sets
    for subgraph_id, subgraphs in all_sets_of_subgraphs.items():
        subgraph[subgraph_id] = {}
        
        # Process each individual subgraph
        for graph in subgraphs:
              
            for node_key, node_list in graph.items(): # Iterate over the nodes in the subgraph
                
                if node_key.startswith('nodes'): # Check if the key is a node key
                    
                    node_key_number = int(node_key[6:])  # Extract digits after 'nodes'
                    last_digit = int(str(node_key_number))  # Get the last digit(s)
                    subgraph[subgraph_id][last_digit] = node_list
    
    subgraphinfo = subgraph[k]
    return subgraph, subgraphinfo


# Function to extract the details of which the task can run on from the application model.
def extract_task_info(JD):
    task_info = {}
    for node in JD["application"]["jobs"]:
        task_info[node["id"]] = node["can_run_on"]
    return task_info



def convert_selInd_to_json(partition, jsondata, folder_name,k):
    extracted_info = {"application": {"jobs":[] , "messages": []}}
    filename = f"{folder_name}/{k}.json"
    if not partition:
        extracted_info["application"]["jobs"].append({
                    "id": 0,
                    "wcet_fullspeed": 0,
                    "mcet": 0,
                    "deadline": 0,
                    "can_run_on": 0,
                    "processing_times": 0
                })
        extracted_info["application"]["messages"].append({
                            "id": 0,
                            "sender": 0,
                            "receiver":0,
                            "size": 0,
                            "period": 0
                        })
    else:
        for nd in partition:
            for job in jsondata["application"]["jobs"]:
                if nd == job["id"]:
                    extracted_info["application"]["jobs"].append({
                        "id": job["id"],
                        "wcet_fullspeed": job["wcet_fullspeed"],
                        "mcet": job["mcet"],
                        "deadline": job["deadline"],
                        "can_run_on": job["can_run_on"],
                        "processing_times": job["processing_times"]
                    })
        
        for message in jsondata["application"]["messages"]:
            if message["sender"] in partition and message["receiver"] in partition:
                extracted_info["application"]["messages"].append({
                            "id": message["id"],
                            "sender": message["sender"],
                            "receiver": message["receiver"],
                            "size": message["size"],
                            "timetriggered": message["timetriggered"],
                            "period": message["period"]
                        })

    
    
    with open(filename, "w") as outfile:
        json.dump(extracted_info, outfile, indent=2)



# Function to remove the cyclic dependencies which may emerge after the addition of new tasks
def find_and_remove_cycles(dependency_dict):
    def dfs(node, visited, rec_stack, path, all_cycles):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in dependency_dict[node]:
            if neighbor not in visited:
                dfs(neighbor, visited, rec_stack, path, all_cycles)
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                if cycle not in all_cycles:
                    all_cycles.append(cycle)
        
        rec_stack.remove(node)
        path.pop()

    def remove_cycle(cycle, dependency_dict):
        # Remove the first edge in the cycle to break it
        if cycle and cycle[0] in dependency_dict and cycle[1] in dependency_dict[cycle[0]]:
            dependency_dict[cycle[0]].remove(cycle[1])
            #print(f"Removed edge from {cycle[0]} to {cycle[1]} to break the cycle: {cycle}")

    while True:
        visited = set()
        rec_stack = set()
        all_cycles = []
        
        for node in dependency_dict:
            if node not in visited:
                dfs(node, visited, rec_stack, [], all_cycles)
        
        if all_cycles:
            #print("Cycles found and removed:")
            for cycle in all_cycles:
                #print(cycle)
                remove_cycle(cycle, dependency_dict)
        else:
            #print("No more cycles found.")
            break

    return dependency_dict

   # Define a function to find the k-shortest paths (using shortest path method)
def k_shortest_paths(G, source, target, k):
    return list(islice(nx.shortest_simple_paths(G, source, target), k))

# A function to calculate path cost based on number of routers in the list
def path_cost(G, path):
    return sum(1 for i in range(1, len(path)))# if G.nodes[path[i]]['is_router'])

# Calculate the k-shortest paths with diversity between two processors
def diverse_k_shortest_paths(G, source, target, k):
    paths = k_shortest_paths(G, source, target, k)
    paths_with_costs = [(path, path_cost(G, path)) for path in paths]
    return paths_with_costs


# Function to find the path cost and the processor found in platform model present in the json files


def load_graph_from_json():
    """Load nodes and edges from JSON files into a NetworkX graph."""
    G = nx.Graph()
    prcr_list = [] # list to store the processor types in the platform model
    unique_edge_pcr = {} # dict to store the raspberry pi and microcontroller nodes details (layers in which they are found)
                    # (the type of processor that is only found in edge layers)
    clocking_speed = {} # dict to store the clocking speed of the processors

    # Detect platform number from application filename (e.g., T2_var_001 -> 2_Platform.json, T20 -> 20_Platform.json)
    import re
    import os
    app_name = cfg.file_name  # e.g., "T2_var_001.json" or "T20.json"
    match = re.match(r'[Tt](\d+)_', app_name)  # Match T followed by digits and underscore (T2_var format)
    if match:
        platform_num = match.group(1)
        pltfile = cfg.platform_dir_path + f"/{platform_num}_Platform.json"
    else:
        # Fallback for non-standard names (T20.json, TNC100.json, etc.)
        # Use Platform 5 as default (has diverse processor types)
        pltfile = cfg.platform_dir_path + "/5_Platform.json"
        if not os.path.exists(pltfile):
            # If Platform 5 doesn't exist, try 3
            pltfile = cfg.platform_dir_path + "/3_Platform.json"
    
    with open(pltfile) as f:
        data = json.load(f)


    #store the raspberry pi and microcontroller nodes details (layers in which they are found) (the type of processor that is only found in edge layers)

    for nd in data["platform"]["nodes"]:
        type_processor = nd.get("type_of_processor", "")
        if type_processor in ['Raspberry Pi 5','Microcontroller','General purpose CPU','FPGA']:
            number = nd["id"]
            if type_processor not in unique_edge_pcr:
                
                unique_edge_pcr[type_processor] = [number]
            else:
                
                unique_edge_pcr[type_processor].append(number)


    # Add nodes
    for node in data["platform"]["nodes"]:
        G.add_node(node["id"], is_router=node["is_router"], type_of_processor=node.get("type_of_processor", ""))
        
        if not node["is_router"]:
            prcr_list.append(node["id"])
        # # Ensure processor_type is added correctly to the dictionary
        # if processor_type:
        #     # Add a dictionary entry for each processor type
        #     if processor_type not in prcr_dict[number]:
        #         prcr_dict[number][processor_type] = []

        #     prcr_dict[number][processor_type].append(node["id"])

    # Add edges
    for link in data["platform"]["links"]:
        G.add_edge(link["start"], link["end"], weight=link.get("weight", 1))  # Assuming weight is stored in the "weight" field

    for node in data['platform']['nodes']:
        if not node['is_router']:  # Only look at processors (not routers)
            processor_id = node['id']
            clk_speed = node.get('clocking_speed', None)  # Extract clocking speed if present

            clocking_speed[processor_id] = clk_speed

    return G, prcr_list,unique_edge_pcr,clocking_speed


# Function to convert clocking speed to Hz
def convert_clocking_speed_to_hz(clocking_speed):
    if 'GHz' in clocking_speed:
        # Convert GHz to Hz (1 GHz = 1e9 Hz)
        return float(re.findall(r"\d+\.\d+|\d+", clocking_speed)[0]) * 1e9
    elif 'MHz' in clocking_speed:
        # Convert MHz to Hz (1 MHz = 1e6 Hz)
        return float(re.findall(r"\d+\.\d+|\d+", clocking_speed)[0]) * 1e6
    else:
        return None
    
# Function to generate paths between processors in a graph and organize them in dictionaries
def generate_processor_paths(processor_nd_pltfm, G, k):
    """
    Generate paths between processors in a graph and organize them in dictionaries.

    Args:
        processor_nd_pltfm (list): List of processors or platforms.
        G (Graph): Graph representing the connections between processors.
        k (int): Number of diverse shortest paths to find.

    Returns:
        tuple: (paths_dict, merged_paths_dict)
            - paths_dict: Dictionary of paths grouped by processor pairs.
            - merged_paths_dict: Flattened dictionary of all paths with combined IDs.
    """
    # Initialize dictionaries to store paths
    paths_dict = {}
    merged_paths_dict = {}

    # Add an empty path for processors connecting to themselves
    merged_paths_dict['00'] = {'path': [0, 0], 'cost': 0}

    # Initialize unique path identifier
    path_id = 1

    # Find paths between all pairs of processors
    for i in range(len(processor_nd_pltfm)):
        for j in range(i + 1, len(processor_nd_pltfm)):
            source = processor_nd_pltfm[i]
            target = processor_nd_pltfm[j]
            paths = diverse_k_shortest_paths(G, source, target, k)

            # Sub-dictionary for this path_id
            sub_paths_dict = {}
            for sub_path_id, (path, cost) in enumerate(paths):
                sub_paths_dict[sub_path_id] = {'path': path, 'cost': cost}

                # Create a merged ID and store the path
                merged_id = f"{path_id}{sub_path_id}"
                merged_paths_dict[merged_id] = {'path': path, 'cost': cost}

            # Add to paths_dict
            paths_dict[path_id] = sub_paths_dict

            # Increment path_id
            path_id += 1

    # Add self-loops for each processor
    for i in range(len(processor_nd_pltfm)):
        source = processor_nd_pltfm[i]
        target = source  # Self-loop

        # Self-loop represented with a cost of 1
        sub_paths_dict = {0: {'path': [source, target], 'cost': 1}}
        paths_dict[path_id] = sub_paths_dict

        # Create a merged ID for the self-loop
        merged_id = f"{path_id}0"
        merged_paths_dict[merged_id] = {'path': [source, target], 'cost': 1}

        # Increment path_id
        path_id += 1

    PathDict = cfg.path_info + "/paths_file.json" 
    # FIX: Use platform-specific paths file to avoid overwriting when switching platforms
    # Extract platform number from cfg.file_name (e.g., T2_var_001 -> 2, T20 -> 20)
    import re
    match = re.match(r'[Tt](\d+)', cfg.file_name)  # Match T followed by digits
    if match:
        platform_num = match.group(1)
        MergedPath = cfg.path_info + f"/Paths_{platform_num}.json"
    else:
        MergedPath = cfg.path_info + "/Paths.json"
        
    with open(PathDict, "w") as paths_out:
            json.dump(paths_dict, paths_out, indent=4)

    with open(MergedPath, "w") as merged_paths_out:
            json.dump(merged_paths_dict, merged_paths_out, indent=4)

    return paths_dict, merged_paths_dict


# Function to find the path mapping , Maps tasks to processors.
    # Updates the message list to include the assigned path index for communication between processors.
    
# def ComputeMappingsAndPathsGlobal(message_list, tasks, processors, message_orderings, path_indices):
#     # Create a deep copy of the message list to avoid modifying the original
#     message_list_copy = deepcopy(message_list)
    
#     #print("message_orderings",message_orderings)
#     # Create a dictionary that maps task IDs to processor IDs
#     task_to_processor = {task: processors[i] for i, task in enumerate(tasks)}
#     #print("task_to_processor",task_to_processor)
#     # Container for the updated message list
#     updated_message_list = []

#     # Map message orderings to their respective path indices
#     message_to_path_mapping = {message_orderings[i]: path_indices[i] for i in range(len(message_orderings))}
#     #print("message_to_path_mapping",message_to_path_mapping)
#     for message in message_list_copy:
#         # Find the path index corresponding to the message ID
#         path_index = message_to_path_mapping.get(message['id'], None)

#         # If sender and receiver are on the same processor, set path_index to 0 (self-loop)
#         if task_to_processor[message['sender']] == task_to_processor[message['receiver']]:
#             path_index = 0

    

#         # Create an updated message structure with the assigned path index
#         updated_message = {
#             'id': message['id'],
#             'sender': task_to_processor[message['sender']],  # Map sender task to processor
#             'receiver': task_to_processor[message['receiver']],  # Map receiver task to processor
#             'size': message['size'],  # Retain message size
#             'path_index': path_index  # Add path index for communication
#         }

#         # Append the updated message to the list
#         updated_message_list.append(updated_message)

#     return updated_message_list


def ComputeMappingsAndPathsGlobal(
    message_list, tasks, processors, message_orderings, path_indices, partition, layer, selected_individual, extracted_tuples, subgraphinfo):
    """
    Computes mappings and paths for a global scheduling problem by updating message paths based on task and processor mappings.

    Args:
        message_list (list): List of messages with sender, receiver, and size information.
        tasks (list): List of task IDs.
        processors (list): List of processor IDs allocated to tasks.
        message_orderings (list): Orderings of messages.
        path_indices (list): Path indices corresponding to partitions.
        partition (list): List of partitions.
        layer (list): List of layers in the architecture.
        selected_individual (dict): Details of the selected individual (specific mapping).
        extracted_tuples (list): Filtered (sender, receiver) tuples.
        subgraphinfo (dict): Mapping of partitions to tasks.

    Returns:
        list: Updated list of filtered messages with assigned path indices and updated structure.
    """
    # Create a deep copy of the message list to avoid modifying the original
    message_list_copy = deepcopy(message_list)

    # Debugging print statements
    # print("Processor Allocation:", processors)
    # print("Tasks:", tasks)   
    # print("Message Orderings:", message_orderings)
    # print("Path Indices:", path_indices)  
    # print("Partition:", partition)
    # print("Layer:", layer)
    # print("Selected Individual:", selected_individual)

    # Map tasks to processors
    task_to_processor = {task: processors[i] for i, task in enumerate(tasks)}
    # print("Task to Processor Mapping:", task_to_processor)

    # Filter messages based on extracted tuples
    filtered_messages = [
        message for message in message_list_copy
        if (message['sender'], message['receiver']) in extracted_tuples
    ]
    # print("Filtered Messages:", filtered_messages)

    # Map partitions to path indices
    message_to_path_mapping = {
        partition[i]: path_indices[i] for i in range(len(partition))
    }
    # print("Message to Path Mapping:", message_to_path_mapping)

    # # Map tasks to partitions for quick lookup
    task_to_partition = {}
    for partition_key, tasks_in_partition in subgraphinfo.items():
        
        for task in tasks_in_partition:
            task_to_partition[task] = partition_key
    # print("Task to Partition Mapping:", task_to_partition)
    # Updated message list with path index
    updated_message_list = []

    for message in filtered_messages:
        sender = message['sender']
        receiver = message['receiver']
        # print("Sender:", sender)
        # print("Receiver:", receiver)
        # Find the partition for the sender
        partition = task_to_partition.get(sender)
        # print("Partition for Sender:", partition)   
        # if partition is not None:
            # Get the path index for the partition
        path_index = message_to_path_mapping.get(partition, None)
        # else:
            # path_index = None  # Handle cases where the partition is not found

        # Check if the sender and receiver are on the same processor
        if task_to_processor[sender] == task_to_processor[receiver]:
            path_index = 0  # Self-loop communication

        # Create an updated message structure with the assigned path index
        updated_message = {
            'id': message['id'],
            'sender': task_to_processor[sender],  # Map sender task to processor
            'receiver': task_to_processor[receiver],  # Map receiver task to processor
            'size': message['size'],  # Retain message size
            'path_index': path_index  # Add path index for communication
        }
        # print("Updated Message:", updated_message)
        # Append the updated message to the list
        updated_message_list.append(updated_message)

    return updated_message_list



def find_suitable_pathsGlobal(updated_message_list, merged_paths_dict):
    """
    Finds suitable paths for messages and returns a dictionary where the sender is the key, 
    and the value is a list containing the receiver and the selected path ID.

    Args:
        updated_message_list (list): List of message dictionaries, each containing 'sender', 'receiver', and 'path_index'.
        merged_paths_dict (dict): Dictionary of precomputed paths with path IDs as keys and details including 'path'.

    Returns:
        dict: Dictionary with sender as keys and values as lists [receiver, path_id].
    """
    # Container for selected paths
    selected_path = {}
    idx=1
    for message in updated_message_list:
        # Extract sender and receiver processor IDs
        sender = message['sender']
        receiver = message['receiver']
        size = message['size']
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
        path_id = next(iter(valid_path_ids), '00')
        comm_cst = merged_paths_dict[path_id]['cost']
        # Add to the dictionary with sender as the key and [receiver, path_id] as the value
        # BUG FIX: Store actual message size, not cost (cost = comm_cst + size was adding delay to message_size)
        # The cost should only be used for path selection logic, not stored in schedule
        
        selected_path[idx] = [sender, receiver, path_id, size]  # Store SIZE not COST
        
        idx = idx + 1
    return selected_path

def update_schedule_with_dependencies(data, task_pair, selected_paths,schdiff):
    # Copy data to avoid modifying the original data
    updated_data = {partition: (tasks.copy(), max_time) for partition, (tasks, max_time) in data.items()}


    # Helper function to recursively update intra-partition dependencies
    def update_intra_partition(partition, task_id, updated_start_time, updated_end_time,schdiff):
        for dependent_task_id, task_data in updated_data[partition][0].items():
            if dependent_task_id == task_id:
                continue

            # Check if this task is dependent on the current task
            for sender, _, _ in task_data[3]:
                if sender == task_id:
                    
                    # Update the dependent task's start and end times
                    task_start, task_end = task_data[1], task_data[2]
                    

                    new_start_time = max(task_start, updated_end_time)
                    if new_start_time - updated_end_time < schdiff.get(partition, {}).get((task_id, dependent_task_id), 0):
                        new_start_time = updated_end_time + schdiff.get(partition, {}).get((task_id, dependent_task_id), 0) 
            
                    new_end_time = new_start_time + (task_end - task_start)
                    
                    # Update the task
                    updated_data[partition][0][dependent_task_id] = (
                        task_data[0], new_start_time, new_end_time, task_data[3]
                    )

                    # Recursively update other tasks
                    update_intra_partition(partition, dependent_task_id, new_start_time, new_end_time,schdiff)

    # Process task pairs and update their dependencies
    # FIX: Track which task pairs have been processed to prevent phantom dependencies
    # The issue: multiple task pairs might map to the same processor pair
    processed_task_pairs = set()
    
    for sender, receiver in task_pair:
        # Skip if we've already processed this exact task pair
        if (sender, receiver) in processed_task_pairs:
            continue
        processed_task_pairs.add((sender, receiver))
        
        sender_partition, receiver_partition = None, None
        # Find partitions containing the sender and receiver tasks
        for partition, (tasks, _) in updated_data.items():
            if sender in tasks:
                sender_partition = partition
            if receiver in tasks:
                receiver_partition = partition
            if sender_partition is not None and receiver_partition is not None:
                break

        # Skip if either task is not found
        if sender_partition is None or receiver_partition is None:
            continue

        # Fetch sender and receiver task details
        sender_data = updated_data[sender_partition][0].get(sender)
    

        receiver_data = updated_data[receiver_partition][0].get(receiver)

        if sender_data and receiver_data:
            # Extract relevant information from Selected_Paths
            # Find the FIRST path matching this task pair's processors
            for path_key, path_info in selected_paths.items():
                sender_proc, receiver_proc, path_id, message_size = path_info
                
                # Match processors for this task pair
                if sender_data[0] == sender_proc and receiver_data[0] == receiver_proc:
                    # Update receiver task start and end time based on message transfer time
                    message_transfer_time = message_size  # Assuming message size represents transfer time
                    
                    # BUG FIX: Check if tasks are on the same node first
                    if sender_proc == receiver_proc:
                        # Same node - no communication delay, task can start immediately after predecessor
                        inter_start_time = sender_data[2]
                        # Set message_size to 0 for same-node communication
                        actual_message_size = 0
                    # Edge to Edge
                    elif 11 <= sender_proc <= 99 and 11 <= receiver_proc <= 99: 
                        inter_start_time = sender_data[2] + message_transfer_time + 50
                        actual_message_size = message_size
                    # Edge to Fog or Fog to Edge
                    elif (11 <= sender_proc <= 99 and 101 <= receiver_proc <= 999) or (101 <= sender_proc <= 999 and 11 <= receiver_proc <= 99):
                        inter_start_time = sender_data[2] + message_transfer_time + 150
                        actual_message_size = message_size
                    # Edge to Cloud or Cloud to Edge 
                    elif (11 <= sender_proc <= 99 and receiver_proc >= 1001) or (sender_proc >= 1001 and 11 <= receiver_proc <= 99):
                        inter_start_time = sender_data[2] + message_transfer_time + 350
                        actual_message_size = message_size
                    # Cloud to Fog or Fog to Cloud
                    elif (101 <= sender_proc <= 999 and receiver_proc >= 1001) or (sender_proc >= 1001 and 101 <= receiver_proc < 999):
                        inter_start_time = sender_data[2] + message_transfer_time + 200
                        actual_message_size = message_size
                    # Cloud to Cloud
                    elif sender_proc >= 1001 and receiver_proc >= 1001:
                        inter_start_time = sender_data[2] + message_transfer_time + 175
                        actual_message_size = message_size
                    else:
                        inter_start_time = sender_data[2] + message_transfer_time
                        actual_message_size = message_size

                     
                    receiver_start_time = max(receiver_data[1], inter_start_time)
                    receiver_end_time = receiver_start_time + (receiver_data[2] - receiver_data[1])
                    
                    updated_data[receiver_partition][0][receiver] = (
                        receiver_data[0], receiver_start_time, receiver_end_time, receiver_data[3] + [(sender, path_id, actual_message_size)]
                    )
                    # Recursively update intra-partition tasks
                    update_intra_partition(receiver_partition, receiver, receiver_start_time, receiver_end_time,schdiff)
                    
                    # Break after finding the first matching path for this task pair
                    # This prevents using multiple paths for the same dependency
                    break

    

    # Ensure tasks without dependencies have start_time = 0
    for partition, (tasks, _) in updated_data.items():
        for task_id, task_data in tasks.items():
            if task_data[3] == []:  # No dependencies
                updated_data[partition][0][task_id] = (
                    task_data[0], 0, task_data[2] - task_data[1], task_data[3]
                )

    # FIX: Resolve processor conflicts - serialize tasks on the same processor across partitions
    # Collect all tasks grouped by processor
    tasks_by_processor = {}
    for partition, (tasks, _) in updated_data.items():
        for task_id, (processor, start_time, end_time, deps) in tasks.items():
            if processor not in tasks_by_processor:
                tasks_by_processor[processor] = []
            tasks_by_processor[processor].append((partition, task_id, start_time, end_time, deps))
    
    # Check each processor for overlaps and serialize if needed
    for processor, task_list in tasks_by_processor.items():
        if len(task_list) <= 1:
            continue  # No overlap possible with single task
        
        # Sort tasks by start time, then by task_id for deterministic ordering
        sorted_tasks = sorted(task_list, key=lambda x: (x[2], x[1]))  # Sort by (start_time, task_id)
        
        # Detect and fix overlaps - multiple passes to handle chains
        max_iterations = len(sorted_tasks) * 2  # Safety limit
        iteration = 0
        changes_made = True
        
        while changes_made and iteration < max_iterations:
            changes_made = False
            iteration += 1
            
            for i in range(len(sorted_tasks) - 1):
                partition_i, task_id_i, start_i, end_i, deps_i = sorted_tasks[i]
                partition_j, task_id_j, start_j, end_j, deps_j = sorted_tasks[i + 1]
                
                # Check for overlap: task i ends after task j starts
                if end_i > start_j:
                    # Overlap detected - push task j to start after task i ends
                    duration_j = end_j - start_j
                    new_start_j = end_i  # Start right after task i ends
                    new_end_j = new_start_j + duration_j
                    
                    # Only update if this actually changes the schedule
                    if new_start_j != start_j:
                        # Update task j in the schedule
                        updated_data[partition_j][0][task_id_j] = (processor, new_start_j, new_end_j, deps_j)
                        
                        # Update the sorted list for subsequent iterations
                        sorted_tasks[i + 1] = (partition_j, task_id_j, new_start_j, new_end_j, deps_j)
                        
                        # Recursively update dependent tasks in the same partition
                        update_intra_partition(partition_j, task_id_j, new_start_j, new_end_j, schdiff)
                        
                        changes_made = True

    return updated_data



# Function to remove duplicates from the schedule
def remove_duplicates_from_schedule(schedule):
    for key, value in schedule.items():
        # Extract the dictionary and the last value (float) from the tuple
        inner_dict, last_value = value
        # Iterate through the inner dictionary
        for inner_key, inner_value in inner_dict.items():
            # Extract the tuple (containing the list of items)
            inner_tuple = inner_value
            # Remove duplicates from the list using a set
            unique_list = list({tuple(item) for item in inner_tuple[3]})
            # Update the inner tuple with the unique list
            inner_dict[inner_key] = (inner_tuple[0], inner_tuple[1], inner_tuple[2], unique_list)
    return schedule


# Function to update the partition times (schedule max time) in schedule 
def update_schedule_times(data):
    # New dictionary to store partition and their maximum times
    max_times = {}
    
    for partition, (tasks, max_time) in data.items():
        # Handle empty schedules (failed partitions)
        if not tasks:
            max_end_time = 999999.0  # High penalty for failed schedule
        else:
            # Find the maximum end time among all tasks in the partition
            max_end_time = max(task_data[2] for task_data in tasks.values())
        
        # Update the partition's max time in the original dictionary
        data[partition] = (tasks, max_end_time)
        
        # Add the maximum time to the new dictionary
        max_times[partition] = max_end_time
    
    return data, max_times

def globalMakespan(data):
    """
    Finds the maximum value in a dictionary and returns the key and value.

    Parameters:
        data (dict): The dictionary with numeric values.

    Returns:
        tuple: A tuple containing the key with the maximum value and the maximum value itself.
    """
    if not data:
        # No partitions - return high penalty
        return 999999.0
    max_partition = max(data, key=data.get)  # Key with the maximum value
    max_value = data[max_partition]         # Maximum value
    return max_value

# # Function to update the individual (partition) by moving tasks between subgraphs to optimisie it further
# def update_individual(connectivity_matrix, subgraphinfo,constrained_task,task_can_run_info):
#     # print("Subgraph Info Before updation :", subgraphinfo)
#     duplicated_subgraphinfo = deepcopy(subgraphinfo)  # Save the original state of subgraphinfo
#     print("Subgraph Info Before updation :", duplicated_subgraphinfo)
#      # Loop to iterate and mutate tasks
#     while True:
#         # Select a random key (subgraph)
#         random_key = random.choice(list(subgraphinfo.keys()))

#         # Check if the selected subgraph has no tasks (empty list), continue if empty
#         if not subgraphinfo[random_key]:
#             continue  # Skip empty subgraphs

#         # Select a random task from the selected subgraph
#         tasks_in_subgraph = random.choice(subgraphinfo[random_key])

#         if tasks_in_subgraph in constrained_task:
#             a = 0
#             continue  # Continue to find another task, skipping the current one as constrained task moving to different partition is not allowed
       

#         # Find adjacent tasks of the selected task based on the connectivity matrix
#         adjacent_tasks = [i for i, val in enumerate(connectivity_matrix[tasks_in_subgraph]) if val == 1]

#         if adjacent_tasks:
#             # Choose a random adjacent task from the list of adjacent tasks
#             random_adjacent_task = random.choice(adjacent_tasks)
#             can_run_adjacent_task = task_can_run_info[random_adjacent_task]
#             can_run_task_in_subgraph = task_can_run_info[tasks_in_subgraph]

#             if any(task in can_run_task_in_subgraph for task in can_run_adjacent_task):
#             # Add the adjacent task to the same or a different subgraph
#                 task_added = False
#                 for key, value in subgraphinfo.items():
#                     if random_adjacent_task in value:
#                         # Move the task to the new subgraph and remove it from the old one
#                         subgraphinfo[key].append(tasks_in_subgraph)  # Move task to the new subgraph
#                         subgraphinfo[random_key].remove(tasks_in_subgraph)  # Remove it from the old subgraph
#                         task_added = True
#                         break

#             # Exit after one iteration as adjacent task has been handled
#             if task_added:
#                 break
#         else:
#             # No adjacent tasks were found, look for an empty subgraph
#             empty_subgraph_key = None
#             for key, value in subgraphinfo.items():
#                 if not value:  # If a subgraph is empty
#                     empty_subgraph_key = key
#                     # print("empty_subgraph_key:", empty_subgraph_key)
#                     break

#             if empty_subgraph_key is not None:
#                 # If an empty subgraph is found, assign the task to it
#                 subgraphinfo[empty_subgraph_key] = [tasks_in_subgraph]  # Assign the task to the empty subgraph
#                 subgraphinfo[random_key].remove(tasks_in_subgraph)  # Remove it from the old subgraph
#                 # print("Subgraph Info After updation of empty key:", subgraphinfo)

#             else:
#                 a = 1
#                 continue  # Skip if no empty subgraphs are found
#             break
#     if subgraphinfo == duplicated_subgraphinfo:
#         print("Same as original")

#     print("Subgraph Info After updation:", subgraphinfo)
#     return subgraphinfo




def update_individual(connectivity_matrix, subgraphinfo, constrained_task, task_can_run_info):
    duplicated_subgraphinfo = deepcopy(subgraphinfo)  # Save the original state
    #print("Subgraph Info Before updation:", duplicated_subgraphinfo)
    max_attempts = cfg.MAX_attempts  # Maximum number of attempts to avoid infinite loop
    attempts = 0  # Track the number of attempts

    while attempts < max_attempts:
        attempts += 1  # Increment attempt count

        # Select a random key (subgraph)
        random_key = random.choice(list(subgraphinfo.keys()))

        # If the selected subgraph is empty, try again
        if not subgraphinfo[random_key]:
            continue

        # Select a random task from the subgraph
        tasks_in_subgraph = random.choice(subgraphinfo[random_key])

        # Skip if task is constrained
        if tasks_in_subgraph in constrained_task:
            continue

        # Find adjacent tasks
        adjacent_tasks = [i for i, val in enumerate(connectivity_matrix[tasks_in_subgraph]) if val == 1]

        if adjacent_tasks:
            # Choose a random adjacent task
            random_adjacent_task = random.choice(adjacent_tasks)
            can_run_adjacent_task = task_can_run_info[random_adjacent_task]
            can_run_task_in_subgraph = task_can_run_info[tasks_in_subgraph]

            if any(task in can_run_task_in_subgraph for task in can_run_adjacent_task):
                # Try to move task to an adjacent task’s subgraph
                task_added = False
                for key, value in subgraphinfo.items():
                    if random_adjacent_task in value:
                        subgraphinfo[key].append(tasks_in_subgraph)  # Move task
                        subgraphinfo[random_key].remove(tasks_in_subgraph)  # Remove from original
                        task_added = True
                        break

                if task_added:
                    break  # Exit loop after a successful move

        # If no valid adjacent task found, try moving it to an empty subgraph
        empty_subgraph_key = next((key for key, value in subgraphinfo.items() if not value), None)

        if empty_subgraph_key is not None:
            subgraphinfo[empty_subgraph_key] = [tasks_in_subgraph]
            subgraphinfo[random_key].remove(tasks_in_subgraph)
            break  # Exit loop after assignment

    # if attempts >= max_attempts:
    #     print("Max attempts reached, stopping to avoid infinite loop.")

    # if subgraphinfo == duplicated_subgraphinfo:
    #     print("Same as original, no changes made.")

    # print("Subgraph Info After updation:", subgraphinfo)
    return subgraphinfo


def convert_partition_to_task_list(partition_dict):
    """
    Converts a partition dictionary to a list where each task's index corresponds to its partition ID.

    Parameters:
        partition_dict (dict): A dictionary where keys are partition IDs and values are lists of tasks.

    Returns:
        list: A list where each index represents a task, and the value at that index is the partition ID.
    """
    # Find the maximum task number to determine the list size
    max_task = max(task for tasks in partition_dict.values() for task in tasks)
    
    # Initialize a list with size equal to the maximum task number + 1
    task_list = [-1] * (max_task + 1)
    
    # Populate the task_list with partition IDs
    for partition_id, tasks in partition_dict.items():
        for task in tasks:
            task_list[task] = partition_id
    
    return task_list