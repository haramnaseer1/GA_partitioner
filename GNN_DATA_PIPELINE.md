# GNN Training Data Generation Pipeline

## Overview
Complete pipeline to generate high-quality GNN training data using the fixed GA (98% validity rate).

## Pipeline Steps

### 1️⃣ Generate All GA Solutions
```bash
python generate_all_ga_solutions.py
```

**What it does:**
- Runs GA on all 107 applications in `Application/` directory
- Validates each solution automatically
- Creates solution files in `solution/*_ga.json`
- Generates detailed report: `ga_generation_report.json`

**Options:**
- `--timeout 300` - Timeout per application (default: 300s)
- `--no-skip` - Regenerate existing solutions
- `--app-dir Application` - Application directory path

**Expected output:**
- ~105 valid solutions (98% success rate)
- 2 invalid (T2_var_016, T2_var_043 - data quality issues)

### 2️⃣ Convert Solutions to Tensors
```bash
python create_tensors.py
```

**What it does:**
- Loads all valid GA solutions
- Extracts graph features (nodes, edges, schedules)
- Converts to PyTorch tensors
- Saves to `training_data.pt`

**Tensor format:**
```python
data_list = [
    {
        'x': tensor([num_nodes, 6]),           # Node features
        'edge_index': tensor([2, num_edges]),  # Edge connectivity
        'edge_attr': tensor([num_edges, 1]),   # Edge features
        'y': tensor([1]),                      # Makespan label
        'num_nodes': int,
        'num_edges': int
    },
    ...
]
```

### 3️⃣ Verify Tensor Data
```bash
python verify_tensors.py
```

**What it does:**
- Loads `training_data.pt`
- Validates tensor shapes and types
- Shows statistics (nodes, edges, makespan distribution)

---

## 🚀 **ONE-COMMAND PIPELINE**

Run everything automatically:
```bash
python prepare_gnn_data.py
```

This will:
1. Clean old solutions (asks for confirmation)
2. Generate all GA solutions with progress tracking
3. Convert to PyTorch tensors
4. Verify data quality
5. Generate complete report

**Estimated time:** 10-15 minutes for 107 applications

---

## GA Bug Fixes Applied

✅ **Bug #1: Platform Detection**
- Fixed hardcoded Platform 5
- Now dynamically detects from filename (T2→Platform 2, T20→Platform 20)

✅ **Bug #2: Fallback Path**
- Fixed incorrect directory reference in platform lookup
- Eliminates FileNotFoundError crashes

✅ **Bug #3: Processor Overlap**
- Added cross-partition overlap detection
- Serializes conflicting tasks automatically

✅ **Bug #4: Phantom Dependencies**
- Prevents duplicate task pair processing
- Eliminates non-existent dependencies in output

**Result:** 39% → 98% validity rate

---

## Output Files

| File | Description |
|------|-------------|
| `solution/*_ga.json` | Individual GA solutions (107 files) |
| `training_data.pt` | PyTorch tensor dataset for GNN training |
| `ga_generation_report.json` | Detailed generation statistics |
| `Logs/global_ga.log` | GA execution logs |

---

## Using the Data for GNN Training

```python
import torch

# Load training data
data_list = torch.load('training_data.pt')

print(f"Total graphs: {len(data_list)}")
print(f"Example graph:")
print(f"  Nodes: {data_list[0]['num_nodes']}")
print(f"  Edges: {data_list[0]['num_edges']}")
print(f"  Makespan: {data_list[0]['y'].item():.2f}")

# Use with PyTorch Geometric
from torch_geometric.data import Data, DataLoader

# Convert to PyG format
pyg_data = [
    Data(
        x=d['x'],
        edge_index=d['edge_index'],
        edge_attr=d['edge_attr'],
        y=d['y']
    )
    for d in data_list
]

# Create data loader
loader = DataLoader(pyg_data, batch_size=32, shuffle=True)

# Train your GNN model
for batch in loader:
    # batch.x, batch.edge_index, batch.edge_attr, batch.y
    ...
```

---

## Troubleshooting

**Q: Some applications fail to generate solutions**
- Check `ga_generation_report.json` for specific errors
- Known issues: T2_var_016, T2_var_043 (incomplete application models)
- 98% success rate is expected

**Q: Tensor conversion fails**
- Ensure solutions exist in `solution/` directory
- Check that application files are in `Application/` directory
- Verify PyTorch is installed: `pip install torch`

**Q: How to regenerate specific applications?**
```bash
# Delete specific solution
Remove-Item solution/T2_var_001_ga.json

# Regenerate
python -m src.main 0 Application/T2_var_001.json
```

---

## Statistics

- **Applications:** 107
- **Expected valid solutions:** 105 (98%)
- **Average GA time:** 5.5s per application
- **Total pipeline time:** 10-15 minutes
- **Tensor dataset size:** ~2-5 MB

---

## Next Steps

1. ✅ Run pipeline: `python prepare_gnn_data.py`
2. ✅ Verify data: Check `ga_generation_report.json`
3. ✅ Train GNN: Use `training_data.pt` in your model
4. 🎯 Evaluate: Compare GNN predictions vs GA solutions
