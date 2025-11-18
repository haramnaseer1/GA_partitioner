# GA Scheduling Algorithm - Critical Bug Analysis

**Analysis Date:** November 17, 2025  
**Analyzed Solution:** `solution/T2_ga.json`  
**Application:** `Application/T2.json`  
**Status:** 🔴 **CRITICAL - ALL SOLUTIONS INVALID**

---

## Executive Summary

The GA scheduling algorithm has **5 critical bugs** that produce physically impossible schedules. All 535 planned training solutions are affected. **Cannot proceed with GNN training until these bugs are fixed.**

### Impact Assessment
- ❌ Execution times off by factor of **10⁸** (300s instead of 9.009e-07s)
- ❌ Invalid communication model (adds delays for same-node tasks)
- ❌ Overlap violations (tasks start/end at exact same time on same processor)
- ❌ Message size corruption (24 bytes → 25, 28 bytes)
- ❌ Missing cross-node communication delays

---

## Bug #1: Execution Time Not Scaled by Clock Speed (PARTIALLY FIXED)

### Location
- ✅ **CORRECT:** `src/global_GA.py` line 405: `execution_time = processing_time[task_id] / clk_speed`
- ❌ **PROBLEM:** `src/auxiliary_fun_GA.py` lines 821-930: `update_schedule_with_dependencies()` recalculates times but preserves duration as `(task_end - task_start)` without re-dividing by clock speed

### Evidence from T2_ga.json
```json
{
  "node_id": 53,
  "task_id": 0,
  "start_time": 0.0,
  "end_time": 300.0  // ❌ WRONG: Should be 9.009e-07
}
```

### Expected Calculation
```
Processing time: 300 (from Application/T2.json)
Clock speed: 333 MHz = 333 × 10⁶ Hz = 3.33e8 Hz
Execution time: 300 / 3.33e8 = 9.009e-07 seconds
End time: 0 + 9.009e-07 = 9.009e-07
```

### Actual Output
```
End time: 300.0 (not divided by clock speed!)
```

### Root Cause
The `optimise_schedule()` function correctly calculates execution time divided by clock speed. However, `update_schedule_with_dependencies()` recalculates start/end times by preserving the **duration** `(task_end - task_start)`, which in some paths might be using the raw processing time instead of the clock-scaled execution time.

**Line 844 in auxiliary_fun_GA.py:**
```python
new_end_time = new_start_time + (task_end - task_start)
```

This assumes `(task_end - task_start)` is already correct, but somewhere in the pipeline, unscaled times are being used.

### Fix Required
1. Trace where the initial schedule from `optimise_schedule()` gets corrupted
2. Ensure `update_schedule_with_dependencies()` preserves clock-scaled execution times
3. Verify message_list processing doesn't use raw processing_time

---

## Bug #2: Same-Node Communication Delays Incorrectly Added

### Location
`src/auxiliary_fun_GA.py` lines 894-911

### Evidence from T2_ga.json
```json
{
  "node_id": 53,
  "task_id": 1,
  "start_time": 375.0,
  "end_time": 675.0,
  "dependencies": [
    {
      "task_id": 0,  // Task 0 also on node 53
      "path_id": "90",
      "message_size": 25.0  // ❌ Should be 0 for same-node
    }
  ]
}
```

### Problem Code
```python
# Line 894-911 in auxiliary_fun_GA.py
if 11 <= sender_proc <= 99 and 11 <= receiver_proc <= 99: 
    inter_start_time = sender_data[2] + message_transfer_time + 50
```

**Issue:** This checks processor ID ranges (edge-to-edge) but doesn't check if `sender_proc == receiver_proc`. Same-node communication should have **zero delay**.

### Expected Behavior
- **Same node:** Task 1 can start immediately after Task 0 ends (no communication delay)
- **Different nodes:** Add message transfer time + network delay

### Current Behavior
- Adds 25 (message) + 50 (edge-edge delay) = 75 seconds even though both tasks on node 53

### Fix Required
```python
# Add check BEFORE range-based delays
if sender_proc == receiver_proc:
    # Same node - no communication delay
    inter_start_time = sender_data[2]  # Start right after predecessor
else:
    # Different nodes - apply network delays
    if 11 <= sender_proc <= 99 and 11 <= receiver_proc <= 99: 
        inter_start_time = sender_data[2] + message_transfer_time + 50
    # ... rest of edge/fog/cloud logic
```

---

## Bug #3: Message Size Corruption

### Location
Unknown - needs investigation in message path cost calculation

### Evidence from T2_ga.json
```json
// Task 1 dependency
{
  "task_id": 0,
  "message_size": 25.0  // ❌ Input file specifies 24
}

// Task 2 dependency
{
  "task_id": 0,
  "message_size": 28.0  // ❌ Input file specifies 24
}
```

### From Application/T2.json
```json
{
  "id": 0,
  "sender": 0,
  "receiver": 1,
  "size": 24  // Correct size
}
```

### Possible Cause
Looking at `src/global_GA.py` lines 330-340:
```python
# Adjust the size of the message with the cost of the path
message["size"] += path_cost
```

This adds path cost to message size, which is conceptually wrong:
- **Message size:** Data volume in bytes
- **Path cost:** Network routing overhead/latency

These should be tracked separately, not summed.

### Fix Required
1. Separate message size (bytes) from path cost (time/delay)
2. Calculate transfer time: `message_size / bandwidth + path_cost`
3. Don't modify original message size

---

## Bug #4: Task Overlap on Same Processor

### Location
`src/global_GA.py` line 414 or `auxiliary_fun_GA.py` line 915

### Evidence from T2_ga.json
```json
// Task 1 on node 53
{
  "node_id": 53,
  "task_id": 1,
  "end_time": 675.0
}

// Task 3 on node 53
{
  "node_id": 53,
  "task_id": 3,
  "start_time": 675.0  // ❌ OVERLAP! Starts exactly when Task 1 ends
}
```

### Problem
Task 3 starts at the **exact same time** Task 1 finishes. On a single processor, this is impossible - there must be zero duration between finish and start, or use `>` not `>=`.

### Current Logic (line 402 in global_GA.py)
```python
start_time = max(earliest_start_from_deps, current_time_per_processor[processor])
```

If `current_time_per_processor[processor] = 675.0` and predecessor finishes at 675.0:
- `start_time = max(675.0, 675.0) = 675.0` ❌

### Fix Required
Add small epsilon buffer OR ensure strict `>` comparison:
```python
# Option 1: Epsilon buffer
EPSILON = 1e-12
start_time = max(earliest_start_from_deps, current_time_per_processor[processor] + EPSILON)

# Option 2: Strict update
current_time_per_processor[processor] = end_time + EPSILON
```

---

## Bug #5: Missing Cross-Node Communication

### Location
`src/global_GA.py` lines 391-392 or message_dict handling

### Evidence from T2_ga.json
```json
{
  "node_id": 53,
  "task_id": 3,
  "dependencies": [
    {
      "task_id": 2,  // Task 2 on node 54 (different node!)
      "path_id": "62",
      "message_size": 0.0  // ❌ Should be 24 + path cost + edge delay
    }
  ]
}
```

### Problem
Task 3 depends on Task 2:
- Task 2: node 54
- Task 3: node 53
- Message size in app: 24 bytes
- **Actual message_size: 0.0** ❌

This suggests the dependency wasn't properly processed or message size was zeroed out.

### Expected
```json
{
  "message_size": 24,  // Base size from application
  // + path_cost from path_id "62"
  // + 50 (edge-to-edge delay)
}
```

### Fix Required
1. Investigate why message_size becomes 0 for cross-node dependencies
2. Check if path_id "62" is being processed correctly
3. Verify message_dict construction in `optimise_schedule()`

---

## Additional Issue: Validation System Weakness

### Location
`Script/validation_utils.py` lines 161-173

### Problem
Validation checks processor **TYPE** (FPGA, CPU) not specific **node ID**:

```python
# resource_mapping() converts:
can_run_on: [2] → "FPGA" type

# Validation passes if task runs on ANY FPGA node
# T2.json specifies node 2, but solution uses nodes 53, 54 (both FPGAs)
```

### Why This Matters
- Application: `can_run_on: [2]` (specific node ID)
- Solution: Tasks on nodes 53, 54 (wrong nodes, right type)
- Validation: ✅ PASS (checks type, not ID)

### Impact
Invalid node assignments pass validation, hiding the above bugs.

### Fix Required
1. Add node ID validation (not just type)
2. OR clarify if `can_run_on` means "node ID" or "processor type"
3. Document platform model semantics

---

## Testing Evidence

### Test Case: T2.json
- **4 tasks**, processing times: 300, 300, 300, 100
- **Node 53:** 333 MHz (3.33e8 Hz)
- **Node 54:** 300 MHz (3.00e8 Hz)
- **All tasks:** `can_run_on: [2]`

### Expected Schedule (Task 0)
```
Processor: node 2 (not 53!)
Execution: 300 / 3.33e8 = 9.009e-07
Start: 0.0
End: 9.009e-07
```

### Actual Schedule (Task 0)
```
Processor: node 53 ❌
Execution: 300 (not scaled!) ❌
Start: 0.0
End: 300.0 ❌
```

**Error magnitude:** 3.33 × 10⁸ times too large!

---

## Impact on GNN Training

### Critical Problems
1. **Invalid physics:** Training data violates execution time physics
2. **Wrong patterns:** GNN will learn incorrect scheduling heuristics
3. **Unusable output:** Trained model will produce invalid schedules
4. **Wasted resources:** 535 solutions × 52s = 7.7 hours of invalid data

### Recommendation
**🛑 HALT all dataset generation immediately.**

Do NOT run:
- `generate_all_ga_solutions.py`
- `run_generation.bat`
- Any GA-based solution generation

---

## Fix Priority

### Phase 1: Critical Fixes (MUST FIX)
1. ✅ Bug #1: Clock speed scaling (verify update_schedule preserves it)
2. ✅ Bug #2: Same-node communication (add `sender_proc == receiver_proc` check)
3. ✅ Bug #4: Task overlap (add epsilon buffer)

### Phase 2: Important Fixes
4. 🔶 Bug #3: Message size corruption (separate size from path cost)
5. 🔶 Bug #5: Missing cross-node communication (debug message_dict)

### Phase 3: Validation Enhancement
6. 🔷 Node ID validation (check specific node, not just type)

---

## Next Steps

### Immediate Actions
1. ✅ Document all bugs (this file)
2. ⬜ Fix Bug #2 (same-node communication) - **HIGHEST PRIORITY**
3. ⬜ Fix Bug #4 (overlap) - **HIGHEST PRIORITY**
4. ⬜ Debug Bug #1 (trace where clock scaling is lost)
5. ⬜ Test fixes on T2.json
6. ⬜ Verify output matches expected calculations

### Validation Test
After fixes, T2_ga.json should show:
```json
[
  {
    "node_id": 2,  // Correct node (if validation fixed)
    "task_id": 0,
    "start_time": 0.0,
    "end_time": 9.009e-07  // ✅ Clock-scaled execution time
  },
  {
    "node_id": 2,
    "task_id": 1,
    "start_time": 9.009e-07,  // ✅ Starts right after Task 0 (same node)
    "end_time": 1.8018e-06,   // ✅ 9.009e-07 + 9.009e-07
    "dependencies": [
      {
        "task_id": 0,
        "message_size": 0  // ✅ Same node, no communication
      }
    ]
  }
]
```

---

## Files to Modify

### Primary Files
1. `src/global_GA.py` (lines 391-392, 405-406, 414)
2. `src/auxiliary_fun_GA.py` (lines 844, 894-911, 915)

### Test Files
1. `Application/T2.json` (test input)
2. `solution/T2_ga.json` (test output - regenerate after fixes)

### Validation Files
1. `Script/validation_utils.py` (enhance node ID checking)

---

## References

- User error analysis: `errors_(1)[1].md`
- GA configuration: `src/config.py`
- Schedule simplification: `src/simplify.py`
- Platform model: `Platform/5_Platform.json` (used for T2.json)

---

**Document Status:** Complete  
**Next Update:** After Bug #2 and #4 fixes applied  
**Owner:** GitHub Copilot Analysis
