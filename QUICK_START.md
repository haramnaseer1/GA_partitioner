# 🚀 Quick Start - Multi-Seed GNN Data Generation

## ONE COMMAND - Generate Everything
```bash
python prepare_gnn_data.py
```
- Will ask: "How many seeds per application?" → Enter **5**
- Generates ~535 training samples in 45-60 minutes
- Creates `training_data.pt` ready for GNN training

---

## Testing First (Optional)
```bash
python test_multiseed.py
```
- Tests with 3 apps × 5 seeds = 15 solutions
- Verifies multi-seed generation works
- Takes ~2 minutes

---

## Manual Control

### Generate Solutions
```bash
# 5 seeds per app (recommended)
python generate_all_ga_solutions.py --seeds 5

# 10 seeds per app (more data)
python generate_all_ga_solutions.py --seeds 10

# Regenerate everything
python generate_all_ga_solutions.py --seeds 5 --no-skip
```

### Convert to Tensors
```bash
python create_tensors.py
```

### Verify Quality
```bash
python verify_tensors.py
```

### Check Status
```bash
python check_pipeline_status.py
```

---

## Output Files

```
solution/
├── T2_ga.json                 # Seed 0
├── T2_seed01_ga.json          # Seed 1
├── T2_seed02_ga.json          # Seed 2
├── T2_seed03_ga.json          # Seed 3
├── T2_seed04_ga.json          # Seed 4
├── T20_ga.json
├── T20_seed01_ga.json
├── ...
└── (535 files total with 5 seeds)

training_data.pt              # PyTorch tensors
ga_generation_report.json     # Statistics
```

---

## Expected Results (5 seeds)

- **Total runs:** 535 (107 apps × 5 seeds)
- **Valid solutions:** ~524 (98% success rate)
- **Training samples:** ~524 graphs
- **Dataset size:** ~10-15 MB
- **Time:** 45-60 minutes

---

## Seed Selection Guide

| Seeds | Samples | Time | Use Case |
|-------|---------|------|----------|
| 3 | ~321 | 25-30 min | Quick test |
| **5** | **~535** | **45-60 min** | **Recommended** ⭐ |
| 10 | ~1070 | 90-120 min | Maximum diversity |
| 20 | ~2140 | 3-4 hours | Research/production |

---

## Troubleshooting

**Q: How do I know if it's working?**
```bash
# Run test first
python test_multiseed.py

# Check status during generation
python check_pipeline_status.py
```

**Q: Can I stop and resume?**
- Yes! Use `--skip-existing` (default) to skip already generated solutions
- Just run the command again

**Q: How to regenerate specific seeds?**
```bash
# Delete specific solutions
Remove-Item solution/T2_var_001_seed03_ga.json

# Regenerate with --no-skip
python generate_all_ga_solutions.py --seeds 5 --no-skip
```

---

## Next Steps

1. ✅ **Generate data:** `python prepare_gnn_data.py`
2. ✅ **Verify:** Check `ga_generation_report.json`
3. ✅ **Train GNN:** Load `training_data.pt`
4. 🎯 **Evaluate:** Compare predictions vs GA solutions

---

**Full Documentation:** See `GNN_SETUP_COMPLETE.md`
