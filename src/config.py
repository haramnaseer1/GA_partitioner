import os
import sys
import datetime
import time
import argparse
import dask
from dask.distributed import Client, LocalCluster, as_completed, wait, Queue as DaskQueue
import multiprocessing as mp
import logging



# ---------Setting the File Name (Application Model Name)  ---------
file_name = 'T2_var_001.json'  # Change the file name to the desired application model name



# ---------------------- Setting the DEBUG_MODE ----------------------
# Setting the DEBUG_MODE

DEBUG_MODE = False  # Change to False to deactivate debug mode (FULL ITERATIONS FOR GNN TRAINING DATA)

# --------------------------------------------------------------------

# --------------------- Setting the operating modes of GA ---------------------
# Setting the Operating Modes of GA

operating_mode = "constrain"  # Change to "constrain" to activate constraint mode else keep it as "nonconstrain"
# "nonconstrain " - In this mode the Application model will be modelled such that all tasks can run on any processor
# "constrain" - In this mode the Application model will be modelled such that some task can run only on specific processors

# --------------------------------------------------------------------

# ---------------------- Setting DASK Mode ----------------------
Parallel_Mode = False  # Change to True to activate DASK Mode   # Change to False to deactivate DASK Mode
cluster = "local"  # for omni cluster mode else keep it as "local" for local cluster mode    

# --------------------------------------------------------------------

# ---------------------- Defining Parameters  ----------------------

# Constants
seed = 42
global_tracking = 0  # Global Tracking Variable for storing the best fitness value, best individual, and best generation

# Subgraph and Layers
num_runs = 10 if not DEBUG_MODE else 3
num_layers = 15 
MAX_attempts = 100 # Maximum number of attempts for modifying the subgraph after fitness evaluation

num_path = 5 # Number of paths that need to be generated

#last_known_time =  1735540178.119759 # Last known time for the platform model modification - Laptop
last_known_time =    1740397401.2023108 # Last known time for the platform model modification - Desktop

# Genetic Algorithm Constants
POPULATION_SIZE_GGA = 20 if not DEBUG_MODE else 3  # Population Size for Global Genetic Algorithm
POPULATION_SIZE_LGA = 10 if not DEBUG_MODE else 3  # Population Size for Local Genetic Algorithm

# For GNN training data: balanced iterations (complete solutions, reasonable time)
# TEMPORARY: Reduced for quick testing (1 hour time limit) - restore to 50/30 for production
NUMBER_OF_GENERATIONS_GCA = 2
NUMBER_OF_GENERATIONS_LGA = 2

MUTATION_PROBABILITY_GGA = 0.4
MUTATION_PROBABILITY_LGA = 0.4

CROSSOVER_PROBABILITY_GGA = 0.8
CROSSOVER_PROBABILITY_LGA = 0.8

TOURNAMENT_SIZE_GGA = 4
TOURNAMENT_SIZE_LGA = 4

indprb = 0.5 # Independent probability for mutation 


#------------------------------------------------------------------------------------------------


# ------- Adding argument parsing to allow dynamic file names -------
# parser = argparse.ArgumentParser(description="Arugumets for the GA, Host ID and file name")
# parser.add_argument('scheduler_host', help="Scheduler hostname")
# #parser.add_argument("file_name", type=str, help="The name of the JSON file")
# args = parser.parse_args()

#---------------------------------------------------------------


# ---------------------- Defining Paths  ----------------------

app_dir_path = "../Application"  # Path for Application Model
platform_dir_path = "../Platform"  # Path for Platform Model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
log_dir_path = "../Logs"  # Path for Logs
fig_dir_path = "../Figures"  # Path for Figures
src_dir_path = "./src"  # Path for Source Code
subgraph_dir_path = "./SubGP_from_DFS_"  # Path for Subgraph generated from DFS algorithm
subGraph_dir_path = "./SubGraphs"  # Path for Subgraphs for GA algorithm selected instance from the DFS algorithm
combined_SubGraph_dir_path = "./Combined_SubGraphs"  # Path for Combined Subgraphs for LGA 
path_info = "./Path_Information"  # Path for Storing the txt file containing the path information in the platform model

#---------------------------------------------------------------



# ---------------------- Function for Printing ----------------------
# Debug Print Function
def debug_print(*args):
    if DEBUG_MODE:
        print(*args)

#---------------------------------------------------------------

# ---------------------- Configuarations ----------------------
# Avoid editing this section

time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_dir = os.path.dirname(os.path.relpath(__file__))
app_dir_path = os.path.join(script_dir, app_dir_path)
platform_dir_path = os.path.join(script_dir, platform_dir_path)
log_dir_path = os.path.join(script_dir, log_dir_path)
fig_dir_path = os.path.join(script_dir, fig_dir_path)
src_dir_path = os.path.join(script_dir, src_dir_path)


# ---------------------- Setting up the logger ----------------------
os.makedirs(log_dir_path, exist_ok=True)
global_ga_logger = logging.getLogger("global_ga_logger")
global_ga_logger.setLevel(logging.INFO)

# Create and configure the file handler
# Check for custom log file path from environment variable
ga_log_file = os.environ.get('GA_LOG_FILE', os.path.join(log_dir_path, "global_ga.log"))
global_ga_handler = logging.FileHandler(ga_log_file)  # Log file
global_ga_formatter = logging.Formatter('%(asctime)s - %(message)s')  # Log format
global_ga_handler.setFormatter(global_ga_formatter)  # Attach formatter to handler

# Avoid duplicate handlers
if not global_ga_logger.handlers:
    global_ga_logger.addHandler(global_ga_handler)