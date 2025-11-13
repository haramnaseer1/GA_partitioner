#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=15          # 15 parallel tasks (one per file)
#SBATCH --cpus-per-task=2    # Assign 2 CPU cores per task (30 cores total)
#SBATCH --mem=200000         # Reduce memory to 200GB (keep some for system)
#SBATCH --partition=long
#SBATCH --time=19-23:00:00
#SBATCH --job-name=GA_100_m

# Load required modules
module load python 

# Activate the virtual environment
source /work/ws-tmp/jp719313-GAGPExp/expgagp/bin/activate

# Set the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/work/ws-tmp/jp719313-GAGPExp/100

# Change to project directory
cd /work/ws-tmp/jp719313-GAGPExp/100


# Step 4: Run the Python script with the scheduler's hostname (which will handle 15 files)
python3 -u -m src.main 0