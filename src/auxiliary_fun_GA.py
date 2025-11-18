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


# Set the Reference Speed for WCET scaling (1 GHz = 1e9 Hz)
REFERENCE_SPEED_HZ = 1e9


# =====================================================================
# TIER MAPPING AND LATENCY CONSTANTS (FROM Guide.pdf §10.7)
# =====================================================================

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
    DELAYS = {
        ('Edge', 'Edge'): 50.0,
        ('Edge', 'Fog'): 150.0,
        ('Fog', 'Edge'): 150.0,
        ('Edge', 'Cloud'): 350.0,
        ('Cloud', 'Edge'): 350.0,
        ('Fog', 'Cloud'): 200.0,
        ('Cloud', 'Fog'): 200.0,
        ('Cloud', 'Cloud'): 175.0,
        ('Fog', 'Fog'): 50.0, 
    }
    return DELAYS.get((sender_tier, receiver_tier), 0.0)


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
    # Create a list to store tasks that can ONLY run on edge devices (Raspberry Pi or Microcontroller)
    # FIX: Only remove tasks that are EXCLUSIVELY constrained to edge devices (3 or 4)
    # Don't remove tasks that CAN run on edge devices but also have other options
    task_ids = []
    
    # Invert the mapping: task_id -> list of resources it can run on
    task_to_resources = {}
    for resource_id, tasks in node_job_mapping.items():
        for task_id in tasks:
            if task_id not in task_to_resources:
                task_to_resources[task_id] = []
            task_to_resources[task_id].append(resource_id)
    
    # Only remove tasks that can ONLY run on edge devices (resources 3 or 4)
    for task_id, resources in task_to_resources.items():
        # Check if task can ONLY run on edge devices (no other options)
        edge_only = all(r in [3, 4] for r in resources)
        if edge_only and task_id not in task_ids:
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
                    # FIX: Only add partitions with at least one task to avoid empty subgraphs
                    if node_list:  # Skip empty partitions
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
    match = re.match(r'[Tt](\d+)_', cfg.file_name)  # Match T#_var format only
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
    
def ComputeMappingsAndPathsGlobal(
    message_list, tasks, processors, message_orderings, path_indices, partition, layer, selected_individual, extracted_tuples, subgraphinfo):
    """
    Computes mappings and paths for a global scheduling problem by updating message paths based on task and processor mappings.
    """
    # Create a deep copy of the message list to avoid modifying the original
    message_list_copy = deepcopy(message_list)

    # Map tasks to processors
    task_to_processor = {task: processors[i] for i, task in enumerate(tasks)}

    # Filter messages based on extracted tuples
    extracted_tuple_set = set(extracted_tuples)
    filtered_messages = [
        message for message in message_list_copy
        if (message['sender'], message['receiver']) in extracted_tuple_set
    ]

    # Map partitions to path indices (not directly used here, but kept for context)
    message_to_path_mapping = {
        partition[i]: path_indices[i] for i in range(len(partition))
    }

    # Map tasks to partitions for quick lookup
    task_to_partition = {}
    for partition_key, tasks_in_partition in subgraphinfo.items():
        
        for task in tasks_in_partition:
            task_to_partition[task] = partition_key
            
    # Updated message list with path index
    updated_message_list = []

    for message in filtered_messages:
        sender = message['sender']
        receiver = message['receiver']
        
        # Find the partition for the sender
        partition = task_to_partition.get(sender)
        
        # Get the path index for the partition (this determines the path choice pool)
        path_index = message_to_path_mapping.get(partition, None)

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
        updated_message_list.append(updated_message)

    return updated_message_list



def find_suitable_pathsGlobal(updated_message_list, merged_paths_dict):
    """
    Finds suitable paths for messages and returns a dictionary where the sender is the key, 
    and the value is a list containing the receiver and the selected path ID.
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

        # Select a valid path ID if found; otherwise, use '00' (empty path cost 0)
        path_id = next(iter(valid_path_ids), '00')
        
        # Add to the dictionary with sender as the key and [receiver, path_id, size] as the value
        selected_path[idx] = [sender, receiver, path_id, size] 
        
        idx = idx + 1
    return selected_path

# Function to update the schedule with dependencies (CRITICAL FIX: Uses Tier Delay Model)
def update_schedule_with_dependencies(data, task_pair, selected_paths,schdiff):
    # Copy data to avoid modifying the original data
    updated_data = {partition: (tasks.copy(), max_time) for partition, (tasks, max_time) in data.items()}


    # Helper function to recursively update intra-partition dependencies
    def update_intra_partition(partition, task_id, updated_start_time, updated_end_time, schdiff, visited=None):
        if visited is None:
            visited = set()

        if task_id in visited:
            return
        visited.add(task_id)

        for dependent_task_id, task_data in updated_data[partition][0].items():
            if dependent_task_id == task_id:
                continue

            # Check if this task is dependent on the current task
            for sender, _, _ in task_data[3]:
                if sender == task_id:
                    
                    # Update the dependent task's start and end times
                    task_start, task_end = task_data[1], task_data[2]
                    
                    new_start_time = max(task_start, updated_end_time)
                    # FIX: Use calculated communication delay (schdiff)
                    if new_start_time - updated_end_time < schdiff.get(partition, {}).get((task_id, dependent_task_id), 0):
                        new_start_time = updated_end_time + schdiff.get(partition, {}).get((task_id, dependent_task_id), 0) 
        
                    new_end_time = new_start_time + (task_end - task_start)
                    
                    # Update the task
                    updated_data[partition][0][dependent_task_id] = (
                        task_data[0], new_start_time, new_end_time, task_data[3]
                    )

                    # Recursively update other tasks
                    update_intra_partition(partition, dependent_task_id, new_start_time, new_end_time, schdiff, visited)
                    # No need to continue loop once dependent task found

    # Process task pairs and update their dependencies
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
            # Find the FIRST path matching this task pair's processors
            for path_key, path_info in selected_paths.items():
                sender_proc, receiver_proc, path_id, message_size = path_info
                
                # Match processors for this task pair
                if sender_data[0] == sender_proc and receiver_data[0] == receiver_proc:
                    
                    # --- CRITICAL FIX: Use Tier-Based Delay Model ---
                    message_transfer_volume = message_size # Traffic volume component
                    
                    if sender_proc == receiver_proc:
                        # Case 1: Same Node (Zero Comm Delay)
                        comm_delay = 0.0
                        actual_message_size = 0.0 # Set to 0.0 for intra-partition communication in schedule output
                    else:
                        # Case 2: Different Nodes (Tier-based fixed delay + message volume)
                        sender_tier = get_tier_from_node_id(sender_proc)
                        receiver_tier = get_tier_from_node_id(receiver_proc)
                        
                        fixed_delay = get_inter_tier_delay(sender_tier, receiver_tier)
                        
                        comm_delay = message_transfer_volume + fixed_delay
                        actual_message_size = message_size # Preserve original size for tracking
                    
                    # Calculate receiver start time
                    inter_start_time = sender_data[2] + comm_delay # Sender End + Total Delay
                    
                    # Resolve processor conflict (inter-partition scheduling relies on the assumption
                    # that the original local schedules handled their own processor non-overlap, 
                    # but inter-partition dependencies might push tasks later)
                    receiver_start_time = max(receiver_data[1], inter_start_time)
                    receiver_end_time = receiver_start_time + (receiver_data[2] - receiver_data[1])
                    
                    # Update receiver schedule entry
                    updated_data[receiver_partition][0][receiver] = (
                        receiver_data[0], receiver_start_time, receiver_end_time, receiver_data[3] + [(sender, path_id, actual_message_size)]
                    )
                    # Recursively update intra-partition tasks
                    update_intra_partition(receiver_partition, receiver, receiver_start_time, receiver_end_time, schdiff)
                    
                    # Break after finding the first matching path for this task pair
                    # This prevents using multiple paths for the same dependency
                    break

    

    # Ensure tasks without dependencies have start_time = 0
    for partition, (tasks, _) in updated_data.items():
        for task_id, task_data in tasks.items():
            if not task_data[3]:  # No dependencies
                # Calculate duration (End - Start)
                duration = task_data[2] - task_data[1]
                updated_data[partition][0][task_id] = (
                    task_data[0], 0, duration, task_data[3]
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
        
        while iteration < max_iterations: # Changed from while changes_made, relying on max_iterations
            changes_made = False
            iteration += 1
            
            for i in range(len(sorted_tasks) - 1):
                partition_i, task_id_i, start_i, end_i, deps_i = sorted_tasks[i]
                partition_j, task_id_j, start_j, end_j, deps_j = sorted_tasks[i + 1]
                
                # Check for overlap: task i ends after task j starts
                # Add epsilon for boundary handling
                if end_i > start_j + 1e-6: 
                    # Overlap detected - push task j to start after task i ends
                    duration_j = end_j - start_j
                    new_start_j = end_i  # Start right after task i ends
                    new_end_j = new_start_j + duration_j
                    
                    # Only update if this actually changes the schedule
                    if abs(new_start_j - start_j) > 1e-6:
                        # Update task j in the schedule
                        updated_data[partition_j][0][task_id_j] = (processor, new_start_j, new_end_j, deps_j)
                        
                        # Update the sorted list for subsequent iterations
                        sorted_tasks[i + 1] = (partition_j, task_id_j, new_start_j, new_end_j, deps_j)
                        
                        # Recursively update dependent tasks in the same partition
                        update_intra_partition(partition_j, task_id_j, new_start_j, new_end_j, schdiff)
                        
                        changes_made = True
            
            if not changes_made and iteration > 1: # Only stop if no changes made after first pass
                break
            elif iteration == max_iterations:
                # print(f"WARNING: Processor {processor} conflict resolution hit max iterations.")
                break

    return updated_data

def globalMakespan(partition_maxtime_Pair):
    """
    Calculates the global makespan from a dictionary of partition makespans.

    Args:
        partition_maxtime_Pair (dict): A dictionary where keys are partition IDs
                                       and values are the makespans of those partitions.

    Returns:
        float: The maximum makespan among all partitions, representing the global makespan.
    """
    if not partition_maxtime_Pair:
        return 0
    return max(partition_maxtime_Pair.values())

def remove_duplicates_from_schedule(schedule):
    """
    Removes duplicate tasks from a schedule dictionary, keeping the entry with the highest end time.
    """
    if not isinstance(schedule, dict):
        return schedule

    cleaned_schedule = {}
    for partition_id, schedule_content in schedule.items():
        # Handle two possible formats: (task_dict, makespan) or just task_dict
        if isinstance(schedule_content, tuple) and len(schedule_content) == 2:
            task_dict, makespan = schedule_content
        else:
            task_dict = schedule_content
            makespan = 0  # Or calculate if needed

        final_tasks = {}
        if isinstance(task_dict, dict):
            for task_id, details in task_dict.items():
                if task_id not in final_tasks or details[2] > final_tasks[task_id][2]:
                    final_tasks[task_id] = details
        
        # Recalculate makespan based on the cleaned tasks
        if final_tasks:
            makespan = max(details[2] for details in final_tasks.values())

        cleaned_schedule[partition_id] = (final_tasks, makespan)

    return cleaned_schedule

def update_schedule_times(schedule):
    """
    Recalculates the makespan for each partition in the schedule based on the maximum task end time.

    Args:
        schedule (dict): The schedule dictionary, where keys are partition IDs and values
                         are tuples of (task_dict, old_makespan).

    Returns:
        tuple: A tuple containing:
            - dict: The updated schedule with corrected makespans.
            - dict: A dictionary mapping each partition ID to its new makespan.
    """
    if not isinstance(schedule, dict):
        return schedule, {}

    updated_schedule = {}
    partition_maxtime_pair = {}

    for partition_id, content in schedule.items():
        if isinstance(content, tuple) and len(content) == 2:
            task_dict, _ = content
        else:
            task_dict = content

        if not task_dict or not isinstance(task_dict, dict):
            new_makespan = 0
        else:
            # Find the maximum end time among all tasks in the partition
            max_end_time = 0
            for task_details in task_dict.values():
                if task_details[2] > max_end_time:
                    max_end_time = task_details[2]
            new_makespan = max_end_time

        updated_schedule[partition_id] = (task_dict, new_makespan)
        partition_maxtime_pair[partition_id] = new_makespan

    return updated_schedule, partition_maxtime_pair

def update_individual(adjacency_matrix, subGraphInfo, constrained_task, task_can_run_info):
    """
    Re-evaluates and updates the subgraph information, particularly for handling non-improving solutions in a GA.
    This function can be adapted to re-run partitioning or scheduling based on the provided context.
    
    For now, it returns the subGraphInfo unchanged, acting as a placeholder.
    A more complex implementation could involve re-running scheduling for each subgraph.
    """
    # Placeholder implementation: returns the original subGraphInfo without modification.
    # This allows the GA to proceed without crashing, though it doesn't add new exploratory behavior.
    # A more advanced version could re-run list scheduling for each partition.
    return subGraphInfo

def convert_partition_to_task_list(subGraphInfo):
    """
    Converts a dictionary of partitions (subGraphInfo) into a single, ordered list of tasks.

    Args:
        subGraphInfo (dict): A dictionary where keys are partition IDs and values are lists of task IDs.

    Returns:
        list: A single list containing all task IDs from the partitions, ordered by partition ID.
    """
    task_list = []
    # Sort by partition ID to ensure deterministic order
    for partition_id in sorted(subGraphInfo.keys()):
        task_list.extend(subGraphInfo[partition_id])
    return task_list