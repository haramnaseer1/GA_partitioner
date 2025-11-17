# Multi-Task GNN for Task Scheduling

Complete pipeline for training GNN models to predict task scheduling solutions.

## Quick Start

### 1. Generate Training Data
```bash
# Full generation (5 seeds, 50/30 iterations, ~535 solutions)
python generate_training_data.py

# Quick test (1 seed, 10/10 iterations)
python generate_training_data.py --quick

# Custom output path
python generate_training_data.py --output data/my_tensors.pt

# Custom configuration
python generate_training_data.py --seeds 3 --gca 40 --lga 25
```

### 2. Train Model
```bash
# Default training (50 epochs, CPU)
python train_model.py

# GPU training with 100 epochs
python train_model.py --epochs 100 --device cuda

# Quick test (10 epochs)
python train_model.py --quick
```

## Commands Reference

### Data Generation

**Basic Usage:**
```bash
python generate_training_data.py [OPTIONS]
```

**Options:**
- `--seeds N` - Number of random seeds per application (default: 5)
- `--gca N` - Global GA iterations (default: 50)
- `--lga N` - Local GA iterations (default: 30)
- `--quick` - Quick test mode (1 seed, 10/10 iterations)
- `--regenerate` - Skip GA, regenerate tensors only
- `--no-skip` - Regenerate all (don't skip existing solutions)
- `--timeout N` - Timeout per GA run in seconds (default: 300)
- `--output PATH` or `-o PATH` - Output file path for tensors (default: training_data_multitask.pt)

**Examples:**
```bash
# Full production run
python generate_training_data.py --seeds 5

# Fast testing
python generate_training_data.py --quick

# Regenerate tensors from existing solutions
python generate_training_data.py --regenerate

# Custom output path
python generate_training_data.py --output data/training_multitask.pt

# Custom iterations for RunPod
python generate_training_data.py --seeds 5 --gca 50 --lga 30
```

### Model Training

**Basic Usage:**
```bash
python train_model.py [OPTIONS]
```

**Options:**
- `--data PATH` - Path to training data (default: training_data_multitask.pt)
- `--epochs N` - Number of epochs (default: 50)
- `--batch-size N` - Batch size (default: 16)
- `--lr FLOAT` - Learning rate (default: 0.001)
- `--device cpu|cuda` - Device to use (default: auto-detect)
- `--quick` - Quick test mode (10 epochs)
- `--resume PATH` - Resume from checkpoint
- `--hidden-dim N` - Hidden dimension (default: 256)
- `--num-layers N` - Number of GAT layers (default: 4)
- `--num-heads N` - Attention heads (default: 8)

**Examples:**
```bash
# Standard training
python train_model.py --epochs 100

# GPU training with custom batch size
python train_model.py --epochs 100 --batch-size 32 --device cuda

# Resume training
python train_model.py --resume models_multitask/best_model.pt --epochs 50

# Quick validation test
python train_model.py --quick
```

## Pipeline Overview

```
Applications (107 files)
         ↓
   [generate_training_data.py]
         ↓
    GA Solutions (535 files)
         ↓
    Multi-task Tensors
         ↓
   [train_model.py]
         ↓
    Trained Model
```

## Output Files

### Data Generation
- `solution/*_ga.json` - GA scheduling solutions
- `training_data_multitask.pt` - PyTorch Geometric training data

### Model Training
- `models_multitask/best_model.pt` - Best model checkpoint
- Training history and metrics in checkpoint

## Model Architecture

**Multi-Task GNN:**
- **Input**: Task graph (nodes: processing_time, deadline, dependencies)
- **Backbone**: 4-layer GAT with multi-head attention
- **Outputs**:
  1. Processor assignment (192-class classification)
  2. Start time (regression)
  3. End time (regression)
  4. Makespan (graph-level regression)

**Parameters**: 1,364,166 trainable parameters

## Data Format

**Input Features (per task):**
- Processing time (float)
- Deadline (float)
- Number of dependencies (int)

**Targets (multi-task):**
- `y_processor`: Processor ID [0-191] (classification)
- `y_start`: Start time in μs (regression)
- `y_end`: End time in μs (regression)
- `y_makespan`: Overall makespan in μs (regression)

## Typical Workflow

### Local Development
```bash
# 1. Quick test
python generate_training_data.py --quick
python train_model.py --quick

# 2. Verify everything works
# Check: training_data_multitask.pt exists
# Check: models_multitask/best_model.pt created
```

### Production (RunPod)
```bash
# 1. Generate full dataset
python generate_training_data.py --seeds 5

# 2. Train full model
python train_model.py --epochs 100 --device cuda --batch-size 32

# 3. Results in models_multitask/best_model.pt
```

## Troubleshooting

**No training data found:**
```bash
python generate_training_data.py
```

**Model training fails:**
- Check `training_data_multitask.pt` exists
- Verify data has multiple graphs (not just 1)
- Try `--quick` mode first

**Out of memory:**
- Reduce `--batch-size`
- Reduce `--hidden-dim`
- Use CPU instead of GPU for small datasets

## Performance Metrics

**Training Progress:**
- Total loss (combined multi-task)
- Processor accuracy (%)
- Start/End time MAE
- Makespan MAE

**Validation:**
- Same metrics on validation set
- Best model saved based on total validation loss

## Bug Fixes Included

✅ **Message Size Bug** (Line 817): Fixed cost+size storage  
✅ **Function Name Bug** (Line 1103): Correct convert_selInd_to_json  
✅ **Unicode Errors**: Fixed console encoding issues  
✅ **Multi-task Architecture**: Complete 4-head model

## Next Steps

After training:
1. Evaluate model on test set
2. Compare predictions with GA baseline
3. Visualize attention weights
4. Export model for deployment

---

**For detailed architecture info**: See `MULTITASK_GNN_ARCHITECTURE.md`
