# A100 Quick Start Guide

## What Changed

Your project has been **optimized for A100 80GB GPU**! Here's what was updated:

### 1. Configuration (config.py)
- ✅ **BATCH_SIZE: 16 → 64** (4x faster training)
- ✅ GPU memory usage: 6% → 30% (better utilization)
- ✅ Expected speedup: **4-5x per model**
- ✅ Full optimization: **~6 days** (was ~25 days)

### 2. Colab Notebook (colab_tutorial.ipynb)
- ✅ Added complete optimization code in **Section 5a**
- ✅ Includes A100 optimizations (TF32, learning rate scaling)
- ✅ Automatic Google Drive checkpoint saving
- ✅ Progress tracking and time estimates

### 3. Helper Scripts
- ✅ `restart_with_a100_optimization.txt` - Clean restart script
- ✅ `A100_OPTIMIZATION_GUIDE.md` - Detailed optimization guide
- ✅ `config_a100.py` - Reference A100 configuration

---

## How to Use in Colab

### Option 1: Start Fresh (Recommended)

Since you only completed 2/24 models in generation 1, I recommend starting fresh with the optimized settings:

1. **Open your Colab notebook**
   - Upload the updated `colab_tutorial.ipynb` to Colab
   - Or clone the repository again to get the latest version

2. **Run cells 1-18** (Setup, mount Drive, verify data)
   - This loads your datasets from Google Drive
   - Verifies everything is working

3. **Backup your old results** (optional)
   ```python
   # Run this in a new cell if you want to backup
   import shutil
   from datetime import datetime

   old_dir = "/content/drive/MyDrive/vindr_optimization"
   backup_dir = f"/content/drive/MyDrive/vindr_optimization_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

   if os.path.exists(old_dir):
       shutil.move(old_dir, backup_dir)
       print(f"✓ Backed up to: {backup_dir}")
   ```

4. **Run Section 5a** (Start New Optimization)
   - This cell now includes all A100 optimizations
   - TF32 will be enabled automatically
   - Batch size is already set to 64
   - Learning rate will be scaled automatically

5. **Let it run!**
   - The optimization will save checkpoints to Google Drive automatically
   - You can resume anytime if the session disconnects
   - Expected time: **~6 days** for full optimization (24 × 50 models)

### Option 2: Continue with Current Batch Size

If you want to continue with batch_size=16:

1. Edit `config.py` in Colab:
   ```python
   # Change this line
   config.BATCH_SIZE = 16  # Change back to 16
   ```

2. Run Section 5b (Resume from checkpoint)

**Note:** This will take **~25 days** total. Not recommended with A100.

---

## What the Optimization Does

When you run **Section 5a**, the code will:

1. **Apply A100 optimizations:**
   ```
   ✓ TF32 enabled for tensor cores
   ✓ Batch size: 64
   ✓ Expected GPU memory: ~19.2 GB / 80 GB
   ✓ Training batches/epoch: ~256
   ✓ Speedup vs batch_size=16: ~4.0x faster
   ✓ Learning rate scaled by 2.0x for larger batches
   ```

2. **Split data:**
   ```
   Training:   16,388 images from 4,000 patients
   Validation: 4,098 images from 1,000 patients
   ```

3. **Start NSGA-III optimization:**
   ```
   Population size: 24
   Generations: 50
   Total models to train: 1,200
   Estimated time per model: ~7 minutes
   Estimated total time: ~5.8 days
   ```

4. **Save checkpoints automatically:**
   - Model checkpoints → `/content/drive/MyDrive/vindr_optimization/checkpoints/`
   - Optimization state → `/content/drive/MyDrive/vindr_optimization/results/`

---

## Monitoring Progress

The optimization will print updates like:

```
Generation 1/50
├── Evaluating individual 1/24... ✓ (PR-AUC: 0.8234, AUROC: 0.9012)
├── Evaluating individual 2/24... ✓ (PR-AUC: 0.8156, AUROC: 0.8987)
...
└── Generation 1 complete! Checkpoint saved.

Time elapsed: 2.8 hours
Estimated remaining: 5.5 days
```

---

## If Session Disconnects

Don't worry! Everything is saved to Google Drive.

1. **Reconnect to Colab**
2. **Run cells 1-18** (remount Drive, load data)
3. **Run Section 5b** (Resume from checkpoint)
   - It will automatically find the latest checkpoint
   - Continue exactly where you left off

---

## Expected Performance

| Metric | Before (batch_size=16) | After (batch_size=64) | Improvement |
|--------|------------------------|----------------------|-------------|
| Time per epoch | ~10-12 min | ~2-3 min | **4-5x faster** |
| Batches per epoch | 1,280 | 256 | **5x fewer** |
| GPU memory | ~5 GB (6%) | ~20 GB (30%) | **4x better utilization** |
| Time per model | ~30 min | ~7 min | **4x faster** |
| Full optimization | ~25 days | **~6 days** | **~19 days saved!** |

---

## Troubleshooting

### Out of Memory Error

If you get CUDA OOM error, reduce batch size:

```python
# Add this before Section 5a
import config
config.BATCH_SIZE = 32  # Try 32 instead of 64
```

### Slow Training

If training is slower than expected:

1. Check GPU type: `!nvidia-smi`
   - Should show "A100-SXM4-80GB"

2. Verify batch size:
   ```python
   import config
   print(f"Batch size: {config.BATCH_SIZE}")
   # Should print: Batch size: 64
   ```

3. Check TF32 is enabled (should be automatic in Section 5a):
   ```python
   import torch
   print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
   print(f"TF32 cudnn: {torch.backends.cudnn.allow_tf32}")
   # Should both print: True
   ```

---

## Summary

✅ **Configuration optimized for A100 80GB GPU**
✅ **Expected speedup: 4-5x faster**
✅ **Total time: ~6 days (was ~25 days)**
✅ **Net savings: ~19 days of compute time!**

**Next step:** Run the updated Colab notebook and start the optimization! 🚀

---

## Questions?

- See `A100_OPTIMIZATION_GUIDE.md` for detailed explanations
- See `RESUME_OPTIMIZATION.md` for resume instructions
- See `restart_with_a100_optimization.txt` for manual restart script
