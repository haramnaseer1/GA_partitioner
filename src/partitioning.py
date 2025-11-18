###########################################################################################
#                                                                                         #
# The file is used to create the initial partitioning of the application.                 # 
# A DFS algorithm is used to create the initial partitions.                               #
#                                                                                         #
###########################################################################################

# Importing the required libraries

from . import config as cfg 
import networkx as nx
import random
import numpy as np
import json
import os
import sys
from src import reading_application_model as ra   # Importing the reading_application_model file

# Function to create the initial partitioning of the application

def dfs(graph, start_node, current_run, current_part):
    """
    Depth-First Search (DFS) to extract a subgraph starting from a given node.
    Args:
        graph: The directed graph to perform DFS on.
        start_node: The node to start the DFS traversal.
        current_run: Identifier for the current run of subgraph generation.
        current_part: Identifier for the current partition in the run.
    Returns:
        subgraph: A dictionary containing nodes and edges of the subgraph.
        dict_key1: Key for the node list in the subgraph dictionary.
        dict_key2: Key for the edge list in the subgraph dictionary.
    """
    visited = [] # List to store the visited nodes
    stack = [start_node] # Stack to store the nodes to be visited/ DFS traversal

    # key to have info of current run and current partition
    dict_key1 = f"nodes{current_run}{current_part}"  
    dict_key2 = f"edges{current_run}{current_part}"
    subgraph = {dict_key1: [], dict_key2: []}

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            # FIX: Traverse both successors AND predecessors to capture full connected component
            # This ensures isolated nodes and backward dependencies are included
            stack.extend(graph.successors(node))
            stack.extend(graph.predecessors(node))
            subgraph[dict_key1].append(node)

            # Extract edges and connections (only outgoing edges)
            for neighbor in graph.successors(node):
                edge = (node, neighbor)
                subgraph[dict_key2].append(edge)

    return subgraph, dict_key1, dict_key2


# Function to generate a set of subgraphs for a given graph
def generate_subgraphs(graph, current_run):
    """
    Generates subgraphs for a given graph, processing constrained and unconstrained tasks separately.
    Args:
        graph: The directed graph to partition.
        current_run: Identifier for the current run of subgraph generation.
    Returns:
        subgraphs: A list of generated subgraphs (both constrained and unconstrained).
    """
    subgraphs = []
    current_part =0 # Current partition number
    max_iterations = len(graph.nodes()) + 10  # Safety limit to prevent infinite loops
    iterations = 0
    
    total_nodes = len(graph.nodes())
    print(f"[Partitioning] Starting with {total_nodes} nodes to partition")
    
    while graph.nodes() and iterations < max_iterations:
        iterations += 1
        current_part =current_part+1
        # Try to start from a node with most connections for better partitioning
        nodes_by_degree = sorted(graph.nodes(), key=lambda n: graph.out_degree(n) + graph.in_degree(n), reverse=True)
        random_start_node = nodes_by_degree[0] if nodes_by_degree else random.choice(list(graph.nodes()))
        
        print(f"[Partitioning] Iteration {iterations}, starting from node {random_start_node}, {len(graph.nodes())} nodes remaining")
        
        result_subgraph,key1,key2 = dfs(graph, random_start_node,current_run,current_part)
        if not result_subgraph[key1]:
            break # Break if no nodes are found in the result.
        
        print(f"[Partitioning] Created partition {current_part} with {len(result_subgraph[key1])} nodes: {result_subgraph[key1]}")
        
        graph.remove_nodes_from(result_subgraph[key1])
        subgraphs.append(result_subgraph)
    
    # Safety check: if graph still has nodes, something went wrong
    if graph.nodes():
        print(f"WARNING: {len(graph.nodes())} nodes not partitioned: {list(graph.nodes())}")
    else:
        print(f"[Partitioning] SUCCESS: All {total_nodes} nodes partitioned into {len(subgraphs)} partition(s)")
    
    return subgraphs

# Function to generate a set of subgraphs for a given graph when constarin mode is active
def generate_subgraphs_constrainmode(graph, current_run,con_g):
    """
    Generates subgraphs for a given graph, processing constrained and unconstrained tasks separately.
    Args:
        graph: The directed graph to partition.
        current_run: Identifier for the current run of subgraph generation.
        constrained_tasks: List of nodes that are constrained to specific processors.
        con_g: Graph containing only constrained tasks.
    Returns:
        subgraphs: A list of generated subgraphs (both constrained and unconstrained).
    """
    subgraphs = []
    current_part =0 # Current partition number
    max_iterations = len(graph.nodes()) + len(con_g.nodes()) + 10
    iterations = 0
    
    total_unconstrained = len(graph.nodes())
    total_constrained = len(con_g.nodes())
    print(f"[Constrain Mode] Starting with {total_unconstrained} unconstrained + {total_constrained} constrained = {total_unconstrained + total_constrained} total nodes")
    
    while (con_g.nodes() or graph.nodes()) and iterations < max_iterations:
        iterations += 1
        # Generate subgraphs for constrained tasks
        while con_g.nodes():
            current_part = current_part+1
            # Prefer high-degree nodes for better partitioning
            nodes_by_degree = sorted(con_g.nodes(), key=lambda n: con_g.out_degree(n) + con_g.in_degree(n), reverse=True)
            random_start_node = nodes_by_degree[0] if nodes_by_degree else random.choice(list(con_g.nodes()))
            
            result_subgraph,key1,key2 = dfs(con_g, random_start_node,current_run,current_part)
            if not result_subgraph[key1]:
                break   # Break if no nodes are found in the result.
            con_g.remove_nodes_from(result_subgraph[key1])
            subgraphs.append(result_subgraph)   
        
        # Generate subgraphs for unconstrained tasks
        while graph.nodes():
            current_part =current_part+1
            # Prefer high-degree nodes for better partitioning
            nodes_by_degree = sorted(graph.nodes(), key=lambda n: graph.out_degree(n) + graph.in_degree(n), reverse=True)
            random_start_node = nodes_by_degree[0] if nodes_by_degree else random.choice(list(graph.nodes()))
            
            print(f"[Constrain Mode] Unconstrained partition {current_part}, starting from node {random_start_node}, {len(graph.nodes())} nodes remaining")
            
            result_subgraph,key1,key2 = dfs(graph, random_start_node,current_run,current_part)
            if not result_subgraph[key1]:
                break # Break if no nodes are found in the result.
            
            print(f"[Constrain Mode] Created unconstrained partition {current_part} with {len(result_subgraph[key1])} nodes: {result_subgraph[key1]}")
            
            graph.remove_nodes_from(result_subgraph[key1])
            subgraphs.append(result_subgraph)
    
    # Safety check
    if graph.nodes() or con_g.nodes():
        remaining = list(graph.nodes()) + list(con_g.nodes())
        print(f"WARNING: {len(remaining)} nodes not partitioned: {remaining}")
    else:
        print(f"[Constrain Mode] SUCCESS: All {total_unconstrained + total_constrained} nodes partitioned into {len(subgraphs)} partition(s)")
    
    return subgraphs

# Function to run the subgraph generation process multiple times in constrained mode
def run_multiple_times_constrainmode(graph, num_runs,constrained_tasks):
    """
    Runs the subgraph generation process multiple times.
    Args:
        graph: The directed graph to partition.
        num_runs: Number of times to repeat the process.
        constrained_tasks: List of nodes constrained to specific processors.
    Returns:
        all_sets_of_subgraphs: A dictionary containing subgraphs for each run.
    """
    all_sets_of_subgraphs = {} # Dictionary to store the subgraphs for each run
    key = 0  # Key to store the subgraphs for each run
    

    for _ in range(num_runs):
        key = key+1
        if cfg.operating_mode == "constrain":
            con_g =nx.DiGraph() # Constrained graph
            con_g.add_nodes_from(constrained_tasks) # Adding the constrained tasks to the constrained graph
            graph_copy = graph.copy()
            subgraphs_for_current_run = generate_subgraphs_constrainmode(graph_copy,key,con_g)
            all_sets_of_subgraphs[key]=subgraphs_for_current_run
        else:
            graph_copy = graph.copy()
            subgraphs_for_current_run = generate_subgraphs(graph_copy,key)
            all_sets_of_subgraphs[key]=subgraphs_for_current_run
    return all_sets_of_subgraphs

# Function to run the subgraph generation process multiple times
def run_multiple_times(graph, num_runs):
    """
    Runs the subgraph generation process multiple times.
    Args:
        graph: The directed graph to partition.
        num_runs: Number of times to repeat the process.
    Returns:
        all_sets_of_subgraphs: A dictionary containing subgraphs for each run.
    """
    all_sets_of_subgraphs = {}
    key =0
    for _ in range(num_runs):
        key = key+1
        graph_copy = graph.copy()
        subgraphs_for_current_run = generate_subgraphs(graph_copy,key)
        all_sets_of_subgraphs[key]=subgraphs_for_current_run
    return all_sets_of_subgraphs




# Function to extract the nodes from the subgraphs for creating the genome.
def extract_nodes_from_subgraphs(all_subgraphs):
    SubGraphs = {}

    for iteration, subpart in enumerate(all_subgraphs.values(), start=1):
        subGinstances = [n[f"nodes{iteration}{i+1}"] for i, n in enumerate(subpart)]
    
        if iteration not in SubGraphs:
            SubGraphs[iteration] = subGinstances
        else:
            SubGraphs[iteration].extend(subGinstances)

    return SubGraphs
    


# Function to extract the message: id, sender, receiver details from the AM used for the original graph 
# coressponding to the subgraph generated and copying to a new json file
                                
def extract_info_of_subgraph(run_key, subgraph, jsondata, current_part, folder_name):
    extracted_info = {"application": {"jobs":[] , "messages": []}}
    edges_key = f"edges{run_key}{current_part}"
    node_key = f"nodes{run_key}{current_part}"
    ed_list = subgraph[edges_key]
    nd_list = subgraph[node_key]
    filename = f"{folder_name}/extracted_infos_{run_key}{current_part}.json"
    for nd in nd_list:
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

    
    for edge in ed_list:
        
        for message in jsondata["application"]["messages"]:
            if edge[0] == message["sender"] and edge[1] == message["receiver"]:
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
    



# Function to create the population for the Genetic Algorithm
# The population is created by assigning each node to the cluster it belongs to
# For example take the node '0', we find the subgraph to which it belongs and assign the subgraph id to the position 0 in the genome
# This is done for all the nodes in the graph
#   
def create_population(subg, total_vertex_jsondata):
    subgp_popn = {}   # dict to store subgraph Population
    for ctl, nodelist in enumerate(subg.values(), start = 1):
        genome= [0 for i in range(total_vertex_jsondata)]
        for cluster_num in range (len(nodelist)):
            for nodeid in nodelist[cluster_num]:   
                genome[nodeid] = cluster_num+1
            subgp_popn[ctl]= genome
    return subgp_popn




