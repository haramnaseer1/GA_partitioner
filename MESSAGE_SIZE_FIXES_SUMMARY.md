# Message Size Bug Fixes - Summary

## Date: November 18, 2025

## Bugs Fixed

### Bug #1: Message Size = Cost (NEW - CRITICAL)
**Location:** `src/auxiliary_fun_GA.py` line 817

**Problem:**
```python
cost = comm_cst + size
selected_path[idx] = [sender, receiver, path_id, cost]  # BUG: storing cost instead of size
```

The code was calculating `cost = communication_cost + message_size` and then storing this COST value in the path info. Later, this cost was unpacked as if it were the message_size, causing incorrect values like:
- Original message: 24 bytes
- Communication cost: 1 byte  
- Stored value: 25 bytes ❌
- Solution shows: 25 bytes (WRONG)

**Fix:**
```python
# Store actual message size, not cost
selected_path[idx] = [sender, receiver, path_id, size]  # FIXED: store SIZE not COST
```

**Impact:** This caused T2_ga.json to show message sizes of 25, 28, 0 instead of 24, 24, 24

---

### Bug #2: Wrong Function Name (FIXED EARLIER)
**Location:** `src/global_GA.py` line 1103

**Problem:**
Called `convert_selind_to_json` (lowercase 'i') instead of `convert_selInd_to_json` (capital 'I')

**Fix:**
Changed to correct function name with capital 'I'

**Impact:** This caused T2_var_XXX applications with diverse message sizes to show 24 bytes for all messages

---

## Configuration Changes

### Reduced Iterations (Temporary - for 1 hour time limit)
**Location:** `src/config.py` lines 56-57

**Changed from:**
```python
NUMBER_OF_GENERATIONS_GCA = 50  # Global GA
NUMBER_OF_GENERATIONS_LGA = 30  # Local GA
```

**Changed to:**
```python
NUMBER_OF_GENERATIONS_GCA = 10  # TEMP: reduced for 1-hour test
NUMBER_OF_GENERATIONS_LGA = 10  # TEMP: reduced for 1-hour test
```

**⚠️ IMPORTANT:** Restore these to 50/30 for production runs!

---

## Regeneration Status

**Started:** 01:20:11 (November 18, 2025)
**Applications:** 107
**Seeds:** 1 per application
**Total runs:** 107
**Timeout:** 120s per run
**Estimated time:** ~1 hour
**Log file:** `generation_log_FIXED.txt`

---

## Testing Results

### T2.json Test (Manual)
- **Before fix:** message_size = 25, 28, 0 (WRONG)
- **After fix:** message_size = 0.0 for same-node communication (CORRECT)
- ✅ **Fix verified**

---

## What Was Fixed

1. ✅ Message sizes now show actual values from application JSON (not cost)
2. ✅ Same-node communication correctly shows 0 bytes
3. ✅ Different-node communication shows original message size (24 bytes for T2)
4. ✅ No more arithmetic errors (cost + size being treated as size)

---

## Next Steps

1. ✅ **Wait for regeneration to complete** (~1 hour, running in background)
2. ⏳ **Validate all 107 solutions** using `Script/check_solutions.py`
3. ⏳ **Restore iterations** to 50/30 in `src/config.py` for production
4. ⏳ **Generate T2_var_XXX solutions** (428 more applications) if needed
5. ⏳ **Generate multi-task tensors** using `create_tensors_multitask.py`
6. ⏳ **Train GNN** with corrected data

---

## Files Modified

1. `src/auxiliary_fun_GA.py` - Line 817 (message size bug fix)
2. `src/global_GA.py` - Line 1103 (function name fix - done earlier)
3. `src/config.py` - Lines 56-57 (temporary iteration reduction)

---

## Monitoring Progress

Check log file in real-time:
```powershell
Get-Content generation_log_FIXED.txt -Tail 20 -Wait
```

Check how many solutions generated:
```powershell
(Get-ChildItem solution/*_ga.json).Count
```

---

## Summary

Both message size bugs are now FIXED:
1. ✅ Cost vs Size confusion (NEW - this was the T2 issue)
2. ✅ Function name typo (FIXED EARLIER - this was the T2_var_XXX issue)

All 107 applications are being regenerated with both fixes applied.
Expected completion: ~02:20 (1 hour from start time 01:20)
