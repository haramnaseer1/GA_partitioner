# 🎯 GNN Training Data Generation - Complete Setup

## 📋 Summary

You now have a **complete pipeline** to generate high-quality GNN training data using the **fixed GA** (98% validity rate) with **multiple random seeds** for data diversity.

---

## 🔧 What's Been Fixed

### GA Bug Fixes (4 Critical Issues)
1. ✅ **Platform Detection** - Dynamic detection from filename (was hardcoded to 5)
2. ✅ **Fallback Path** - Corrected directory reference in platform lookup
3. ✅ **Processor Overlap** - Cross-partition overlap detection and serialization
4. ✅ **Phantom Dependencies** - Task pair deduplication to prevent non-existent deps

**Result:** Validity improved from 39% → 98%

---

## 🎲 Multi-Seed Generation

**NEW:** Generate multiple solutions per application using different random seeds!

- **107 applications** × **5 seeds** = **~535 training samples**
- Each seed produces a different (valid) solution
- More data diversity → Better GNN learning
- Customizable: Use 3-10 seeds per application

---

## 📁 Pipeline Files Created

| File | Purpose |
|------|---------|
| `prepare_gnn_data.py` | **ONE-COMMAND** pipeline (recommended) |
| `generate_all_ga_solutions.py` | Generate GA solutions for all apps |
| `create_tensors.py` | Convert solutions to PyTorch tensors |
| `verify_tensors.py` | Verify tensor data quality |
| `check_pipeline_status.py` | Check current status |
| `GNN_DATA_PIPELINE.md` | Complete documentation |

---

## 🚀 How to Generate Training Data

### Option 1: ONE COMMAND (Recommended)
```bash
python prepare_gnn_data.py
```

This will automatically:
1. Clean old solutions (asks confirmation)
2. **Ask for number of seeds** (default: 5)
3. Generate all GA solutions with multiple seeds
4. Convert to PyTorch tensors
5. Verify data quality
6. Generate detailed report

**Time:** 
- 5 seeds × 107 apps = ~535 solutions in ~45-60 minutes
- 10 seeds × 107 apps = ~1070 solutions in ~90-120 minutes

---

### Option 2: Step by Step
```bash
# Step 1: Generate solutions with 5 seeds per app
python generate_all_ga_solutions.py --seeds 5

# Step 2: Convert to tensors
python create_tensors.py

# Step 3: Verify
python verify_tensors.py

# Check status anytime
python check_pipeline_status.py
```

**Seed Options:**
- `--seeds 3` - Quick test (107 × 3 = ~321 samples)
- `--seeds 5` - Good balance (107 × 5 = ~535 samples) ⭐ **Recommended**
- `--seeds 10` - Maximum diversity (107 × 10 = ~1070 samples)

---

## 📊 Expected Output

### With 5 Seeds per Application:

**Files Generated:**
- `solution/*_ga.json` - 535 solution files total
  - `T2_ga.json`, `T2_seed01_ga.json`, ..., `T2_seed04_ga.json`
  - `T20_ga.json`, `T20_seed01_ga.json`, ..., `T20_seed04_ga.json`
  - Same pattern for all 107 applications
- `training_data.pt` - PyTorch tensor dataset (~10-15 MB)
- `ga_generation_report.json` - Detailed statistics

### Statistics:
- **Applications:** 107
- **Seeds per app:** 5 (customizable)
- **Total solutions:** ~535 (107 × 5 × 98% validity)
- **Valid solutions:** ~524 (98% success rate)
- **Avg GA time:** 5.5s per run
- **Total time:** 45-60 minutes

### File Naming:
- Seed 0: `T2_var_001_ga.json`
- Seed 1: `T2_var_001_seed01_ga.json`
- Seed 2: `T2_var_001_seed02_ga.json`
- Seed 3: `T2_var_001_seed03_ga.json`
- Seed 4: `T2_var_001_seed04_ga.json`

---

## 🎓 Using the Data for GNN Training

### Load Training Data:
```python
import torch

# Load dataset
data_list = torch.load('training_data.pt')

print(f"Total graphs: {len(data_list)}")
# Output: Total graphs: 105

# Example graph structure
example = data_list[0]
print(f"Nodes: {example['num_nodes']}")
print(f"Edges: {example['num_edges']}")
print(f"Makespan: {example['y'].item():.2f}")
```

### Tensor Structure:
```python
{
    'x': tensor([num_nodes, 6]),           # Node features
    'edge_index': tensor([2, num_edges]),  # Edge connectivity (COO format)
    'edge_attr': tensor([num_edges, 1]),   # Edge features
    'y': tensor([1]),                      # Graph label (makespan)
    'num_nodes': int,                      # Number of nodes
    'num_edges': int                       # Number of edges
}
```

### Node Features (6 features):
1. `processing_time` - Task processing time
2. `deadline` - Task deadline
3. `start_time` - Scheduled start time
4. `end_time` - Scheduled end time
5. `processor_id` - Assigned processor
6. `is_scheduled` - 1.0 if scheduled, 0.0 otherwise

### Edge Features (1 feature):
1. `message_size` - Communication size between tasks

---

## 📈 Scaling Options

### Current Setup: 107 Applications

**Option 1: More Seeds (Recommended)**
- 5 seeds: ~535 samples ⭐
- 10 seeds: ~1070 samples
- 20 seeds: ~2140 samples

**Option 2: More Applications**
- Add more JSON files to `Application/` directory
- Follow naming: `T{platform}_var_{id}.json`
- Run pipeline again

**Time Estimates:**
| Configuration | Solutions | Time |
|--------------|-----------|------|
| 107 apps × 3 seeds | ~321 | 25-30 min |
| 107 apps × 5 seeds | ~535 | 45-60 min ⭐ |
| 107 apps × 10 seeds | ~1070 | 90-120 min |
| 200 apps × 5 seeds | ~1000 | 90 min |

---

## 🔍 Validation & Quality Assurance

### Automatic Validation:
Every solution is validated for:
- ✅ **Precedence constraints** - Task dependencies satisfied
- ✅ **Non-overlap constraints** - No processor conflicts
- ✅ **Eligibility constraints** - Tasks on compatible processors

### Generation Report:
Check `ga_generation_report.json` for:
```json
{
  "total": 107,
  "valid_count": 105,
  "invalid_count": 2,
  "failed_count": 0,
  "valid": ["T2_var_001", "T2_var_002", ...],
  "invalid": [
    {"app": "T2_var_016", "reason": "..."},
    {"app": "T2_var_043", "reason": "..."}
  ],
  "total_time_seconds": 589.2,
  "avg_time_seconds": 5.5
}
```

---

## 🛠️ Troubleshooting

### Q: Some applications fail?
**A:** Check `ga_generation_report.json` for specific errors. Expected: 98% success rate.

### Q: How to regenerate specific apps?
```bash
# Delete solution
Remove-Item solution/T2_var_001_ga.json

# Regenerate
python -m src.main 0 Application/T2_var_001.json
```

### Q: How to regenerate all solutions?
```bash
python generate_all_ga_solutions.py --no-skip
```

### Q: Tensor conversion fails?
**A:** Ensure:
- Solutions exist in `solution/` directory
- Applications exist in `Application/` directory
- PyTorch installed: `pip install torch`

---

## 📝 Next Steps

1. **Generate Data:**
   ```bash
   python prepare_gnn_data.py
   ```

2. **Review Report:**
   ```bash
   python check_pipeline_status.py
   cat ga_generation_report.json
   ```

3. **Train GNN Model:**
   - Load `training_data.pt`
   - Use PyTorch Geometric
   - Train on graph scheduling task

4. **Evaluate:**
   - Compare GNN predictions vs GA solutions
   - Measure makespan accuracy
   - Test generalization on new applications

---

## 📚 Documentation

- **Full Pipeline Guide:** `GNN_DATA_PIPELINE.md`
- **GA Fixes Summary:** `GA_FIXES_SUMMARY.md`
- **Generation Scripts:** All `.py` files in root directory

---

## ✅ Current Status

- ✅ GA bugs fixed and validated (98% validity)
- ✅ Complete pipeline scripts created
- ✅ Tensor conversion ready
- ✅ Documentation complete
- 🎯 **READY TO GENERATE TRAINING DATA**

---

## 🚀 Quick Start

```bash
# Generate all training data in one command
python prepare_gnn_data.py
```

That's it! ✨
