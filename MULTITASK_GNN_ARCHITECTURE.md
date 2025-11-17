# Multi-Task GNN Architecture

## Problem Analysis

The previous single-task model (`SchedulingGNN` in `train_gnn_scheduling.py`) **cannot** handle multi-task data because:

### Old Model (Single-Task)
- **Input**: Task graph features (node features, edges)
- **Output**: Single value - Makespan prediction `[batch_size, 1]`
- **Architecture**: Shared backbone → Single regression head
- **Parameters**: 1,158,020

### New Data Requirements (Multi-Task)
The multi-task dataset (`training_data_multitask.pt`) requires:

**Node-level predictions (per task):**
1. **Processor Assignment** - Classification (192 classes)
   - Target: `y_processor` `[num_nodes]`
   - Task: Predict which processor executes each task
   
2. **Start Time** - Regression
   - Target: `y_start` `[num_nodes]`
   - Task: Predict when each task starts
   
3. **End Time** - Regression
   - Target: `y_end` `[num_nodes]`
   - Task: Predict when each task completes

**Graph-level prediction:**
4. **Makespan** - Regression
   - Target: `y_makespan` `[batch_size]`
   - Task: Predict overall schedule completion time

---

## Solution: Multi-Task GNN

### New Model Architecture (`MultiTaskSchedulingGNN`)

```
Input: Task Graph
  ├─ Node features: [num_nodes, 3] (processing_time, deadline, dependencies)
  ├─ Edge index: [2, num_edges] (task dependencies)
  └─ Edge attr: [num_edges, 1] (message sizes)
       │
       ▼
┌──────────────────────────────────────┐
│   SHARED ENCODER & BACKBONE          │
├──────────────────────────────────────┤
│ • Node Encoder (3 → 256)             │
│ • Edge Encoder (1 → 64)              │
│ • 4x GAT Layers (multi-head attn)    │
│ • Batch Normalization                │
│ • Residual Connections               │
└──────────────────────────────────────┘
       │
       ├─────────────┬─────────────┬─────────────┬──────────────┐
       ▼             ▼             ▼             ▼              │
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌────────────────┐  │
│  HEAD 1   │ │  HEAD 2   │ │  HEAD 3   │ │    HEAD 4      │  │
│ Processor │ │Start Time │ │ End Time  │ │   Makespan     │  │
│           │ │           │ │           │ │                │  │
│ Node-lvl  │ │ Node-lvl  │ │ Node-lvl  │ │  Graph-lvl     │  │
│Classifier │ │ Regressor │ │ Regressor │ │   Regressor    │  │
│           │ │           │ │           │ │                │  │
│[N, 192]   │ │ [N, 1]    │ │ [N, 1]    │ │ [B, 1]         │  │
└───────────┘ └───────────┘ └───────────┘ └────────────────┘  │
                                                                │
Output Dictionary: {                                           │
  'processor': [num_nodes, 192],  # Logits for 192 processors  │
  'start_time': [num_nodes, 1],   # Start time per task        │
  'end_time': [num_nodes, 1],     # End time per task          │
  'makespan': [batch_size, 1]     # Overall makespan           │
}                                                               │
```

### Key Features

**1. Shared Backbone (Feature Extraction)**
- Learns unified task representation
- Captures task dependencies via attention
- Processes message sizes as edge features
- 4 GAT layers with residual connections

**2. Task-Specific Heads**
Each head specialized for different prediction:

| Head | Type | Output Shape | Loss Function |
|------|------|-------------|---------------|
| Processor | Classification | [N, 192] | CrossEntropyLoss |
| Start Time | Regression | [N, 1] | MSELoss / L1Loss |
| End Time | Regression | [N, 1] | MSELoss / L1Loss |
| Makespan | Regression | [B, 1] | MSELoss / L1Loss |

**3. Multi-Task Learning Benefits**
- **Shared representations**: Common patterns learned once
- **Better generalization**: Multiple objectives prevent overfitting
- **Richer predictions**: Complete scheduling solution (not just makespan)
- **Client requirements**: Directly outputs processor assignment + timing

---

## Model Statistics

### Parameters
- **Total**: 1,364,166 trainable parameters
- **Shared backbone**: ~800K parameters
- **Task heads**: ~564K parameters

### Memory Requirements
- **Model size**: ~5.2 MB (FP32)
- **Training batch**: ~20-50 MB per batch (depends on graph sizes)

---

## Implementation

### File: `train_gnn_multitask.py`

**Key Components:**

```python
# Model creation
model = MultiTaskSchedulingGNN(
    node_feature_dim=3,      # [processing_time, deadline, num_dependencies]
    edge_feature_dim=1,      # [message_size]
    hidden_dim=256,
    num_gat_layers=4,
    num_heads=8,
    num_processors=192,      # 192 compute nodes in platform
    dropout=0.2
)

# Forward pass
outputs = model(data)  # Returns dict with 4 predictions

# Access predictions
processor_logits = outputs['processor']   # [num_nodes, 192]
start_times = outputs['start_time']       # [num_nodes, 1]
end_times = outputs['end_time']           # [num_nodes, 1]
makespan = outputs['makespan']            # [batch_size, 1]
```

### Training Requirements

**Multi-task loss function:**
```python
# Weighted combination of all losses
total_loss = (
    w1 * CrossEntropyLoss(processor_pred, y_processor) +
    w2 * MSELoss(start_pred, y_start) +
    w3 * MSELoss(end_pred, y_end) +
    w4 * MSELoss(makespan_pred, y_makespan)
)
```

**Metrics to track:**
- Processor accuracy (classification)
- Start/End time MAE (mean absolute error)
- Makespan MAE
- Overall multi-task loss

---

## Data Pipeline

### Current Data
- **Old**: `training_data.pt` (535 graphs, makespan-only)
  - Compatible with: `SchedulingGNN` (single-task)
  
- **New**: `training_data_multitask.pt` (1 graph currently, needs regeneration)
  - Compatible with: `MultiTaskSchedulingGNN` (multi-task)
  - **Issue**: Most GA solutions are corrupt/incomplete
  - **Solution**: Regenerate all solutions on RunPod with bug fixes

### Next Steps

1. **Regenerate GA solutions** (RunPod with bug fixes)
   - Run `generate_all_ga_solutions.py` on all 107 applications
   - Expected: ~535-1000 valid solutions
   
2. **Create multi-task tensors**
   - Run `create_tensors_multitask.py` on valid solutions
   - Output: `training_data_multitask.pt`
   
3. **Create training script** 
   - Build `train_multitask_main.py`
   - Implement multi-task loss and metrics
   
4. **Train model**
   - Start with small batch for testing
   - Full training on RunPod GPU
   
5. **Evaluate & Deploy**
   - Test all 4 prediction heads
   - Compare with GA baseline
   - Export for inference

---

## Validation Tests

### Model Architecture Test ✅
```bash
python train_gnn_multitask.py
```
**Result**: Model created successfully with 1,364,166 parameters

### Forward Pass Test ✅
**Input**: 5 nodes, 4 edges  
**Output Shapes**:
- Processor logits: `[5, 192]` ✅
- Start time: `[5, 1]` ✅
- End time: `[5, 1]` ✅
- Makespan: `[1, 1]` ✅

---

## Comparison Summary

| Feature | Single-Task (Old) | Multi-Task (New) |
|---------|------------------|------------------|
| File | `train_gnn_scheduling.py` | `train_gnn_multitask.py` |
| Outputs | 1 (makespan) | 4 (processor, start, end, makespan) |
| Parameters | 1,158,020 | 1,364,166 |
| Data Format | `training_data.pt` | `training_data_multitask.pt` |
| Loss Type | Single MSE | Multi-task (CE + 3×MSE) |
| Client Needs | ❌ Partial | ✅ Complete |

---

## Conclusion

**The old model CANNOT handle multi-task data** because:
1. It only has 1 output head (makespan regression)
2. It cannot predict processor assignments (classification)
3. It cannot predict node-level timing (start/end times)

**The new `MultiTaskSchedulingGNN` solves this** by:
1. Adding 4 specialized output heads
2. Supporting both classification (processors) and regression (times)
3. Providing both node-level and graph-level predictions
4. Meeting all client requirements for scheduling predictions

**Next action**: Create training script for multi-task model, then regenerate clean data on RunPod.
