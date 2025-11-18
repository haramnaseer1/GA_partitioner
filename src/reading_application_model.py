###########################################################################################
#                                                                                         #
# The file reads the application model from the json file                                 #
#                                                                                         #
###########################################################################################

# Importing the required libraries
#---------------------------------
import json
import os
import sys
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from . import config as cfg
import argparse

# Reading the application model from the json file

# FIX: Don't cache file_name - read from cfg.file_name directly to pick up command line changes
# file_name = cfg.file_name

def read_application_model():
    with open(os.path.join(cfg.app_dir_path, cfg.file_name)) as f:
        app_model = json.load(f)
    return app_model



# Function to extract the information needed from the Json Files for the Application Model
####--------------------------------------------------------------------------------------------------------------------####


# Finding the sender and receiver pairs in the Application Model where key is the center and value is the receivers
def find_sender_receiver_pairs(AM):
    sender_receiver_pair = {}
    for sd_re in AM["application"]["messages"]:
        sender = sd_re["sender"]
        receiver = sd_re["receiver"]
        if sender not in sender_receiver_pair:
            sender_receiver_pair[sender] = [receiver]
        else:
            sender_receiver_pair[sender].append(receiver)
    return sender_receiver_pair


# Finding the processing time of the tasks in the Application Model where key is the task and value is the processing time
def find_processing_time(AM):
    processing_cost = {}
    for process in AM["application"]["jobs"]:
        proc_node = process["id"]
        proc_cost = process["processing_times"]
        processing_cost[proc_node] = proc_cost
    return processing_cost



# Function to find the sender receiver pairs as tuples , this for plotting the graph at later stages
def find_sender_receiver_pairs_tuple(sender_receiver_pair):
    vertex_edge_pairs = []

    for source, targets in sender_receiver_pair.items():
        for target in targets:
            vertex_edge_pairs.append((source, target))
    return vertex_edge_pairs

# Function to find the communication cost between the sender and receiver pairs in the Application Model, 
# Here we consider the communication cost the message size parameter defined in the application model Json File
def find_communication_cost(AM):
    messages = AM['application']['messages']
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

# Function to find the message list (a dict to store the messages attributes of the tasks in the Application Model)
# The message list is a dictionary where the key is the task and the value is the list of messages
def find_message_list(AM):
    messages = AM['application']['messages']
    task_ids = [task['id'] for task in AM['application']['jobs']]
    message_list = []

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


# Function to plot the graph of the Application Model
# The graph is plotted using the networkx library and the matplotlib library for visualization and it is a directed graph
def plot_graph(vertex_edge_pairs):
    G = nx.DiGraph()
    for v,e in vertex_edge_pairs:
        G.add_edge(v,e)
   
    pos = nx.spring_layout(G, k=30)
    plt.figure(figsize=(18, 15))
    if cfg.DEBUG_MODE:
        nx.draw(G, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', arrowsize=10)
        # plt.show()  # Commented out to prevent blocking during batch validation
    return G

# Function to find the ajacency matrix of the Application Model(Graph)
def find_adjacency_matrix(vertex_edge_pairs):
    # Find the maximum node value to determine the size of the matrix
    max_node = max(max(pair) for pair in vertex_edge_pairs)
    
    # Initialize a square matrix filled with zeros
    connect_matrix = np.zeros((max_node + 1, max_node + 1), dtype=int)

    # Set values in the matrix based on the vertex-edge pairs
    for edge in vertex_edge_pairs:
        connect_matrix[edge[0], edge[1]] = 1

    conn_m= connect_matrix.T
    
    return conn_m

