# GA Partitioner Delivery Folder Guide

This guide provides a quick overview of the key files included in your delivery folder and their purpose. Use this as a reference to understand the project structure and how to get started with documentation and experiments.

## Main Files

- **README_THESIS.md**
  - Comprehensive thesis guide covering the Genetic Algorithm (GA), tensor usage, and Graph Neural Network (GNN) model architecture. Start here for background, methodology, and technical details.

- **predict_schedule.py**
  - Inference script for generating schedules using the trained GNN model. Run this file to predict task schedules from application JSON files. The script prints the predicted schedule and makespan, and saves results to a JSON file.

- **train_gnn_constrained.py**
  - Training script for the constraint-aware GNN model. Use this to train or retrain the model on new data. Includes logic for saving progress and handling interruptions.

- **requirements.txt**
  - List of required Python packages. Install dependencies using `pip install -r requirements.txt` before running any scripts.

- **Application/**
  - Contains example application JSON files (e.g., `T2_var_001.json`). Use these as input for inference or training.

- **models/**
  - Directory for storing trained model weights (e.g., `gnn_model_constrained.pth`).

## How to Use

1. **Install Dependencies**
   - Open a terminal in this folder and run: `pip install -r requirements.txt`

2. **Run Inference**
   - Use `predict_schedule.py` to generate schedules from application files:
     ```
     python predict_schedule.py Application/T2_var_001.json
     ```
   - The output schedule and makespan will be printed and saved as a JSON file.

3. **Train the Model**
   - If you need to retrain the GNN, use `train_gnn_constrained.py`.

4. **Documentation**
   - Refer to `README_THESIS.md` for a detailed explanation of the project, algorithms, and model architecture.

## Notes
- The GNN model may still be training; stopping early can result in suboptimal results.
- Data generation using GA is time-intensive and may affect overall project timelines.

For further details, consult the code files and thesis guide. If you have questions, please reach out for support.
