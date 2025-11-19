# GNN Schedule Inference Guide

## Overview
This guide explains how to use the GNN-based inference script to predict task schedules for your applications. The script takes an application JSON file as input and generates a predicted schedule that respects constraints like task dependencies, processor eligibility, and timing.

## Prerequisites
- Python 3.12 or higher
- Required packages: PyTorch, PyTorch Geometric, and others listed in `requirements.txt`
- Trained GNN model (automatically loaded by the script)

## Quick Start

### 1. Prepare Your Application File
Ensure your application data is in JSON format, similar to the examples in the `Application/` folder. The file should contain:
- `jobs`: List of tasks with IDs, processing times, deadlines, and eligible processors
- `messages`: Communication dependencies between tasks

Example file: `Application/T2_var_001.json`

### 2. Run the Inference Script
Open a terminal in the project directory and run:

```bash
python predict_schedule.py Application/T2_var_001.json
```

Replace `T2_var_001.json` with your application file name.

### 3. View the Results
The script will:
- Load the trained GNN model
- Process your application data
- Generate a predicted schedule
- Print a summary of the schedule to the console
- Save the full schedule to a JSON file (e.g., `T2_var_001_gnn_schedule.json`)

## Output Format
The predicted schedule JSON contains an array of task assignments:

```json
[
  {
    "task_id": 0,
    "node_id": 172,
    "start_time": 0.0,
    "end_time": 1.5,
    "dependencies": [
      {"task_id": 1, "message_size": 24}
    ]
  }
]
```

- `task_id`: Unique task identifier
- `node_id`: Assigned processor (0-191)
- `start_time` / `end_time`: Predicted execution window
- `dependencies`: Tasks this one depends on, with message sizes

## Troubleshooting
- **Model Loading Errors**: Ensure the trained model file exists in `models/gnn_model_constrained.pth`
- **Invalid Application Format**: Check your JSON matches the expected structure
- **Dependency Issues**: Verify all required packages are installed via `pip install -r requirements.txt`

## Notes
- The GNN respects constraints like precedence, processor eligibility, and communication delays
- Predictions are based on learned patterns from training data
- For best results, use applications similar to those used in training

If you encounter issues, check the console output for error messages or contact the development team.