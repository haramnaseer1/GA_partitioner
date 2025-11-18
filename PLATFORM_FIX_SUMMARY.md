# Platform Mismatch Bug Fixes

## Root Cause Analysis

### Problem
65 out of 107 GA solutions (61%) were invalid due to constraint violations:
- **54 cases:** Eligibility violations (tasks assigned to "Unknown" processor)
- **11 cases:** Non-overlap violations

### Root Cause
**The GA was hardcoded to use Platform 5 for ALL applications**, regardless of their intended platform.

#### Evidence
1. **File:** `src/global_GA.py`, Line 1117
   ```python
   # FIX: Force all partitions to use Platform 5 to avoid processor mismatch issues
   assigned_layer = 5  # Force Platform 5 for all partitions
   ```

2. **Impact:**
   - Application `T2_var_003` should use Platform 2 (processors 21-26)
   - GA used Platform 5 (processors 51-54) instead
   - Solution contained processors `[53, 54]` from Platform 5
   - Validator checked Platform 2 → processors 53, 54 not found → "Unknown" type
   - Result: **Eligibility violation**

3. **Why This Happened:**
   - Original code assigned partitions to "layers" in a multi-tier architecture
   - Someone hardcoded layer/platform 5 as a "fix" for processor mapping conflicts
   - This broke the platform detection logic that should use `T#_` pattern

## Fixes Applied

### Fix 1: Platform Detection in GA (global_GA.py)
**Location:** Lines 1110-1129

**Before:**
```python
assigned_layer = 5  # Force Platform 5 for all partitions
```

**After:**
```python
# Detect platform from application filename
import re
app_name = cfg.file_name  # e.g., "T2_var_003.json"
match = re.match(r'[Tt](\d+)_', app_name)
if match:
    platform_model_str = match.group(1)  # e.g., "2" for T2_var_003
else:
    # Fallback for non-standard names
    platform_model_str = "5"
    if not os.path.exists(os.path.join(cfg.platform_dir_path, f"{platform_model_str}_Platform.json")):
        platform_model_str = "3"

assigned_layer = platform_model_str
```

**Effect:**
- `T2_var_*` apps → Platform 2 (processors 21-26)
- `T3_var_*` apps → Platform 3 (processors 31-34)
- `T5_var_*` apps → Platform 5 (processors 51-54)
- Non-matching apps → Platform 5 (fallback)

### Fix 2: Platform Fallback Path (auxiliary_fun_GA.py)
**Location:** Line 499

**Before:**
```python
pltfile = cfg.path_info + "/3_Tier_Platform.json"  # WRONG directory!
```

**After:**
```python
pltfile = cfg.platform_dir_path + "/3_Platform.json"
```

**Effect:**
- Fallback now uses correct directory (`Platform/` not `Path_Information/`)
- Prevents FileNotFoundError for non-standard application names

## Expected Impact

### Eligibility Violations (54 cases → 0 expected)
- **Before:** Tasks assigned to processors not in platform file → "Unknown" type
- **After:** Tasks assigned to processors from correct platform file → Valid types

### Non-overlap Violations (11 cases → TBD)
- **Hypothesis:** May have been caused by platform mismatch
- **Testing needed:** Re-run GA and validate

### Overall Solution Quality
- **Before:** 39% valid (42/107)
- **Target:** >90% valid
- **Next step:** Re-generate all solutions and validate

## Testing Plan

1. **Single Application Test:**
   - Run GA on `T2_var_003.json`
   - Verify solution uses Platform 2 processors (21-26)
   - Validate solution with `check_solutions.py`
   - Expected: 100% constraints satisfied

2. **Batch Test (10 applications):**
   - Run GA on sample of applications (T2_var_001-010)
   - Validate all solutions
   - Measure: eligibility pass rate, overlap pass rate

3. **Full Validation (107 applications):**
   - Re-run GA on all 107 applications
   - Use `validate_all.py`
   - Compare results with baseline

## Rollback Plan

If the fix causes issues, revert by:
1. Restore line in `global_GA.py`: `assigned_layer = 5`
2. Restore line in `auxiliary_fun_GA.py`: `pltfile = cfg.path_info + "/3_Tier_Platform.json"`

## Files Modified

1. `src/global_GA.py` - Lines 1110-1129 (platform detection logic)
2. `src/auxiliary_fun_GA.py` - Line 499 (fallback path fix)
