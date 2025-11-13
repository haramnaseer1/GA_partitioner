# GA Partitioner - GNN Scheduler Training System

A Genetic Algorithm (GA) based task scheduling system for multi-tier edge-fog-cloud platforms, designed to generate ground-truth training data for Graph Neural Network (GNN) schedulers.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Pipeline](#workflow-pipeline)
- [Data Formats](#data-formats)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [GNN Training Pipeline](#gnn-training-pipeline)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This system solves the **task scheduling problem** on heterogeneous multi-tier platforms using Genetic Algorithms, then uses these solutions as training data for Graph Neural Networks to learn intelligent scheduling policies.

### Key Features

- ✅ **Multi-tier Platform Support**: Edge (11-99), Fog (101-999), Cloud (≥1001) hierarchy
- ✅ **Heterogeneous Resources**: 6 processor types (CPU, FPGA, RPi5, MCU, HPC, GPU)
- ✅ **GA-based Scheduling**: Global + Local GA with partitioning strategy
- ✅ **GNN Training Data**: 107 application benchmarks with optimal schedules
- ✅ **Preprocessing Pipeline**: Feature normalization and encoding
- ✅ **Extensible Architecture**: Easy to add new applications or platforms

### Problem Formulation

**Input:**
- Task DAG (Directed Acyclic Graph) with dependencies
- Platform topology (nodes, links, processor types)
- Resource constraints (can_run_on, processing times)

**Output:**
- Task-to-Node mapping
- Start/end times for each task
- Optimized makespan (total execution time)

**Objective:**
- Minimize makespan
- Respect task dependencies
- Satisfy resource constraints

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Application Models                          │
│         (DAG with tasks, messages, constraints)              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│           GA Partitioner & Scheduler                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐              │
│  │  DFS     │→ │ Global   │→ │   Local GA   │              │
│  │Partition │  │    GA    │  │  (parallel)  │              │
│  └──────────┘  └──────────┘  └──────────────┘              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Scheduling Solutions                            │
│         (107 optimized schedules in JSON)                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│            Preprocessing Pipeline                            │
│  • Normalize features (times, speeds)                        │
│  • Encode resources & tiers                                  │
│  • Create graph tensors                                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              GNN Training (Future)                           │
│  • Graph Convolutional Networks                              │
│  • Supervised learning from GA solutions                     │
│  • Fast inference for runtime scheduling                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
GA_Partitioner/
├── Application/              # Task DAG models (107 applications)
│   ├── T2.json              # Simple 4-task example
│   ├── T20.json             # 20-task benchmark
│   ├── T2_var_001.json      # Variant applications
│   └── ...                  # (100+ more variants)
│
├── Platform/                 # Platform topology models
│   └── EdgeAI-Trust_Platform.json  # 3-node FPGA platform
│
├── solution/                 # GA scheduling outputs (107 files)
│   ├── T2_ga.json           # Optimized schedule for T2
│   └── ...                  # One per application
│
├── gnn_solution/            # Preprocessed GNN training data
│   ├── preprocessed_data.pkl      # Normalized dataset
│   ├── feature_stats.json         # Normalization statistics
│   ├── graph_dataset.pt           # PyTorch Geometric graphs
│   ├── train_test_split.pt        # Train/test split (85/22)
│   └── dataset_info.json          # Graph dataset metadata
│
├── src/                      # Core GA implementation
│   ├── main.py              # Entry point & orchestration
│   ├── global_GA.py         # Global genetic algorithm
│   ├── partitioning.py      # DFS-based graph partitioning
│   ├── List_Schedule.py     # List scheduling heuristic
│   ├── reading_application_model.py  # JSON parsers
│   ├── config.py            # Configuration & parameters
│   └── ...                  # Supporting modules
│
├── Script/                   # Automation & preprocessing
│   ├── generate_all_gnn_data.py   # Batch GA execution
│   ├── preprocess_gnn_data.py     # Feature normalization
│   └── convert_to_gnn_tensors.py  # PyTorch Geometric conversion
│
├── Logs/                     # Execution logs
│   ├── global_ga.log        # GA execution details
│   └── gnn_data_generation.log    # Batch processing log
│
├── Path_Information/         # Network routing paths
├── Combined_SubGraphs/       # Temporary partitioned graphs
├── SubGraphs/               # DFS-generated subgraphs
└── README.md                # This file
```

---

## 🔧 Installation

### Prerequisites

- **Python**: 3.12 or higher
- **OS**: Windows, Linux, or macOS
- **Memory**: 8GB+ recommended for large applications

### Dependencies

```bash
# Core dependencies
pip install deap dask[complete] plotly numpy

# For GNN training (Task 1.3+)
pip install torch torch-geometric
```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd GA_Partitioner

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import deap, dask, plotly; print('✅ All dependencies installed')"
```

---

## 🚀 Quick Start

### 1. Generate Single Solution

```bash
# Edit configuration
cd GA_Partitioner
# Set file_name in src/config.py to your application

# Run GA scheduler
python -m src.main 0

# Generate solution JSON
python -m src.simplify --input Application/T2.json

# Check output
cat solution/T2_ga.json
```

### 2. Generate All Solutions (Batch)

```bash
# Generate solutions for all 107 applications
python Script/generate_all_gnn_data.py

# Monitor progress
tail -f Logs/gnn_data_generation.log
```

### 3. Preprocess for GNN Training

```bash
# Normalize features and encode data
python Script/preprocess_gnn_data.py

# Output: gnn_solution/preprocessed_data.pkl
#         gnn_solution/feature_stats.json
```

### 4. Convert to Graph Tensors

```bash
# Create PyTorch Geometric graph tensors
python Script/convert_to_gnn_tensors.py

# Output: gnn_solution/graph_dataset.pt
#         gnn_solution/train_test_split.pt
#         gnn_solution/dataset_info.json
```

---

## 🔄 Workflow Pipeline

### Stage 1: Data Generation (✅ Complete)

```bash
# Step 1: Run GA on all applications
python Script/generate_all_gnn_data.py
# → Generates 107 solution/*.json files

# Step 2: Preprocess for GNN
python Script/preprocess_gnn_data.py
# → Creates gnn_solution/preprocessed_data.pkl

# Step 3: Convert to graph tensors
python Script/convert_to_gnn_tensors.py
# → Creates gnn_solution/graph_dataset.pt
#          gnn_solution/train_test_split.pt
```

### Stage 2: GNN Training (Next Steps)

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# Load dataset
splits = torch.load('gnn_solution/train_test_split.pt')
train_loader = DataLoader(splits['train'], batch_size=16, shuffle=True)
test_loader = DataLoader(splits['test'], batch_size=16)

# Define GNN model
class SchedulerGNN(torch.nn.Module):
    def __init__(self, num_node_features=10, hidden_dim=64, num_classes=3):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train loop
model = SchedulerGNN(num_node_features=10, hidden_dim=64, num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch).argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    accuracy = correct / total
    print(f'Epoch {epoch}: Loss={total_loss:.4f}, Test Acc={accuracy:.4f}')
```

---

## 📊 Data Formats

### Application Model (Input)

```json
{
  "application": {
    "jobs": [
      {
        "id": 0,
        "processing_times": 300,
        "can_run_on": [2],        // Resource type IDs
        "deadline": 10000
      }
    ],
    "messages": [
      {
        "id": 0,
        "sender": 0,
        "receiver": 1,
        "size": 24                 // Message size in bytes
      }
    ]
  }
}
```

### Platform Model

```json
{
  "platform": {
    "nodes": [
      {
        "id": 51,
        "type_of_processor": "FPGA",
        "clocking_speed": "333 MHz",
        "is_router": false
      }
    ],
    "links": [
      {"start": 51, "end": 52}
    ]
  }
}
```

### Scheduling Solution (Output)

```json
[
  {
    "node_id": 52,                // Assigned node
    "task_id": 1,                 // Task identifier
    "start_time": 379.0,          // Execution start
    "end_time": 679.0,            // Execution end
    "dependencies": [             // Communication times
      {
        "task_id": 0,
        "path_id": "51342",
        "message_size": 29.0
      }
    ]
  }
]
```

### Graph Tensor Format

Each graph in the dataset is a `torch_geometric.data.Data` object:

```python
Data(
    x=[num_tasks, 10],              # Task node features
    edge_index=[2, num_edges],      # DAG connectivity
    edge_attr=[num_edges, 2],       # Message features
    y=[num_tasks],                  # Node assignments (labels)
    platform_x=[num_platform_nodes, 4],  # Platform features
    platform_ids=[num_platform_nodes],   # Platform node IDs
    num_tasks=int,                  # Metadata
    num_messages=int,
    app_name=str
)
```

**Node Features (x):** [10 dimensions]
- `[0]` processing_time_norm
- `[1]` deadline_norm
- `[2]` in_degree (dependencies)
- `[3]` out_degree (successors)
- `[4-9]` can_run_on (one-hot, 6 resource types)

**Edge Features (edge_attr):** [2 dimensions]
- `[0]` message_size_norm
- `[1]` message_size_kb_norm

**Platform Features (platform_x):** [4 dimensions]
- `[0]` clock_speed_norm
- `[1]` resource_type_id (1-6)
- `[2]` tier (0=Edge, 1=Fog, 2=Cloud)
- `[3]` is_router (0/1)

**Labels (y):**
- Platform node ID where each task is assigned

---

## ⚙️ Configuration

### Main Configuration (`src/config.py`)

```python
# Application selection
file_name = 'T2.json'

# GA parameters
POPULATION_SIZE_GGA = 4           # Global GA population
POPULATION_SIZE_LGA = 2           # Local GA population
NUMBER_OF_GENERATIONS_GCA = 4     # Global generations
NUMBER_OF_GENERATIONS_LGA = 4     # Local generations

# Mutation/Crossover
MUTATION_PROBABILITY_GGA = 0.4
CROSSOVER_PROBABILITY_GGA = 0.8

# Debug mode
DEBUG_MODE = False                # Set True for verbose output

# Parallel execution
Parallel_Mode = False             # Enable DASK parallelization
```

### Resource Type Mapping

```python
RESOURCE_TYPES = {
    1: 'General purpose CPU',
    2: 'FPGA',
    3: 'Raspberry Pi 5',
    4: 'Microcontroller',
    5: 'High Performance CPU',
    6: 'GPU'
}
```

### Tier Classification

```python
# Based on node ID ranges
Edge:  11 ≤ node_id ≤ 99
Fog:   101 ≤ node_id ≤ 999
Cloud: node_id ≥ 1001
```

---

## 💡 Usage Examples

### Example 1: Run GA on Custom Application

```bash
# 1. Create your application JSON
cat > Application/my_app.json << EOF
{
  "application": {
    "jobs": [
      {"id": 0, "processing_times": 100, "can_run_on": [1,2,5]},
      {"id": 1, "processing_times": 200, "can_run_on": [2,6]}
    ],
    "messages": [
      {"id": 0, "sender": 0, "receiver": 1, "size": 50}
    ]
  }
}
EOF

# 2. Update config
# Edit src/config.py: file_name = 'my_app.json'

# 3. Run GA
python -m src.main 0

# 4. Generate solution
python -m src.simplify --input Application/my_app.json

# 5. View result
cat solution/my_app_ga.json
```

### Example 2: Analyze Feature Statistics

```python
import json

# Load feature statistics
with open('gnn_solution/feature_stats.json') as f:
    stats = json.load(f)

print(f"Processing time range: {stats['feature_stats']['processing_time']['min']} - {stats['feature_stats']['processing_time']['max']}")
print(f"Clock speed range: {stats['feature_stats']['clock_speed']['min']} - {stats['feature_stats']['clock_speed']['max']} MHz")
print(f"Total samples: {stats['metadata']['num_samples']}")
```

### Example 3: Load Graph Tensors for GNN Training

```python
import torch
from torch_geometric.loader import DataLoader

# Load full dataset
data = torch.load('gnn_solution/graph_dataset.pt')
all_graphs = data['graphs']

# Or load train/test split
splits = torch.load('gnn_solution/train_test_split.pt')
train_graphs = splits['train']  # 85 graphs
test_graphs = splits['test']    # 22 graphs

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

# Inspect a batch
for batch in train_loader:
    print(f"Batch size: {batch.num_graphs}")
    print(f"Node features: {batch.x.shape}")      # [total_nodes, 10]
    print(f"Edge indices: {batch.edge_index.shape}")  # [2, total_edges]
    print(f"Labels: {batch.y.shape}")             # [total_nodes]
    break
```

---

## 🧠 GNN Training Pipeline

### Data Flow

```
Application DAG → GA Scheduler → Solution → Preprocessor → GNN
     (input)       (optimizer)    (label)   (features)   (model)
```

### Feature Engineering

**Node Features (per task):**
- Processing time (normalized)
- Deadline (normalized)
- Resource constraints (one-hot encoded)
- Task degree (in/out edges)

**Edge Features (per message):**
- Message size (normalized)
- Sender/receiver IDs
- Path latency

**Platform Features (per node):**
- Clock speed (normalized)
- Resource type (categorical: 1-6)
- Tier (categorical: 0=Edge, 1=Fog, 2=Cloud)
- Router flag (binary)

**Labels (ground truth):**
- Node assignment (which platform node executes task)
- Start time (normalized)
- End time (normalized)

### Training Objective

```python
# Supervised learning: predict node assignment
loss = CrossEntropyLoss(predicted_node, actual_node)

# Multi-task: also predict timing
loss += MSELoss(predicted_time, actual_time)
```

---

## 🔍 Troubleshooting

### Common Issues

**Issue: "FileNotFoundError: Combined_SubGraphs/*.json"**
```bash
# Solution: Ensure global_GA.py uses relative paths
# Already fixed in current version - uses os.path.join()
```

**Issue: "ModuleNotFoundError: No module named 'src'"**
```bash
# Solution: Run from GA_Partitioner directory
cd GA_Partitioner
python -m src.main 0  # NOT: python src/main.py
```

**Issue: GA timeouts on large applications**
```bash
# Solution: Reduce population size in config.py
POPULATION_SIZE_GGA = 2  # Instead of 4
DEBUG_MODE = True        # For verbose logging
```

**Issue: Solution file not created**
```bash
# Debug steps:
1. Check Logs/global_ga.log for errors
2. Verify application JSON is valid
3. Ensure platform model exists
4. Check disk space for solution/
```

### Performance Optimization

```python
# Enable parallel processing
Parallel_Mode = True
cluster = "local"

# Reduce generations for faster execution
NUMBER_OF_GENERATIONS_GCA = 2
NUMBER_OF_GENERATIONS_LGA = 2

# Increase for better solutions
NUMBER_OF_GENERATIONS_GCA = 10
```

---

## 📈 Project Status

### ✅ Completed (Stage 1: Data Preparation)

- [x] GA scheduler implementation
- [x] DFS-based graph partitioning
- [x] Global & local GA optimization
- [x] Batch processing for 107 applications
- [x] Solution generation (100% success rate)
- [x] Feature preprocessing & normalization
- [x] Resource type & tier encoding
- [x] **PyTorch Geometric tensor conversion**
- [x] **Graph dataset creation (107 samples)**
- [x] **Train/test split (85/22)**

### 🚧 Ready for Stage 2

- [ ] GNN model implementation (GCN/GAT/GraphSAGE)
- [ ] Training pipeline with DataLoader
- [ ] Loss functions & optimization
- [ ] Validation metrics (accuracy, makespan)


## 🤝 Contributing

### Adding New Applications

```bash
# 1. Create application JSON
# 2. Follow schema in Application/T2.json
# 3. Run GA: python -m src.main 0
# 4. Generate solution: python -m src.simplify --input Application/your_app.json
```

### Adding New Platforms

```json
// Create Platform/your_platform_Platform.json
{
  "platform": {
    "nodes": [
      {"id": 11, "type_of_processor": "FPGA", "clocking_speed": "500 MHz"}
    ],
    "links": [{"start": 11, "end": 12}]
  }
}
```

## 📚 References

### Key Algorithms

1. **Genetic Algorithm**: DEAP library (Distributed Evolutionary Algorithms in Python)
2. **Graph Partitioning**: DFS-based subgraph decomposition
3. **List Scheduling**: Priority-based task ordering

### Academic Context

This system implements concepts from:
- Multi-tier edge computing
- Heterogeneous task scheduling
- Graph Neural Networks for combinatorial optimization
- Supervised learning from algorithmic solutions

### Platform Model

- **Edge Tier**: Low-power devices (RPi5, MCU)
- **Fog Tier**: Mid-range compute (FPGA, CPU)
- **Cloud Tier**: High-performance (HPC, GPU)

Communication delays modeled as tier-dependent constants:
- Edge-Edge: 50ms
- Edge-Fog: 150ms
- Fog-Cloud: 200ms
- Edge-Cloud: 350ms
- Fog-Fog: 175ms

