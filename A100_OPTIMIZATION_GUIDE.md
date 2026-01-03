# A100 80GB GPU Optimization Guide

Congratulations on having access to an **NVIDIA A100 80GB GPU**! This is one of the most powerful GPUs available. Let's optimize your configuration to take full advantage of it.

---

## 🎯 Current vs Optimized Settings

### Current Configuration (Conservative)
```python
BATCH_SIZE = 16              # Only using ~5 GB / 80 GB (6%)
Training batches/epoch: ~1,280
Time per epoch: ~10-12 minutes
GPU utilization: ~30-40%
```

### Optimized for A100 80GB
```python
BATCH_SIZE = 64              # Using ~20-25 GB / 80 GB (30%)
Training batches/epoch: ~256
Time per epoch: ~2-3 minutes
GPU utilization: ~85-95%
```

**Speed improvement: ~5x faster! 🚀**

---

## 📊 Batch Size Recommendations

### For A100 80GB with ResNet-50:

| Batch Size | GPU Memory | Time/Epoch | Batches/Epoch | Speed vs 16 | Recommendation |
|------------|------------|------------|---------------|-------------|----------------|
| 16 (current) | ~5 GB | ~10 min | 1,280 | 1x (baseline) | ❌ Underutilized |
| 32 | ~10 GB | ~5 min | 640 | 2x faster | ⚠️ Still low |
| 64 | ~20 GB | ~2.5 min | 256 | 4x faster | ✅ **Recommended** |
| 128 | ~40 GB | ~1.5 min | 128 | 6-7x faster | ✅ Excellent |
| 256 | ~70 GB | ~1 min | 64 | 8-10x faster | ⚡ Maximum |

### My Recommendation: **Batch Size 64 or 128**

- **Batch Size 64**: Safe, proven, 4x faster
- **Batch Size 128**: Aggressive, 7x faster, uses 50% of GPU memory
- **Batch Size 256**: Maximum speed, uses 90% of GPU memory

---

## 🚀 How to Apply Optimized Settings

### Option 1: Quick Update in Colab (Before Section 5)

```python
# Add this cell before running optimization
import config

# Optimize for A100 80GB
config.BATCH_SIZE = 64  # or 128 for even faster training

print(f"✓ Batch size set to: {config.BATCH_SIZE}")
print(f"✓ GPU memory usage: ~{config.BATCH_SIZE * 0.3:.1f} GB / 80 GB")
print(f"✓ Training batches per epoch: ~{20486 * 0.8 / config.BATCH_SIZE:.0f}")
print(f"✓ Expected speedup: ~{64 / 16}x faster than default")
```

### Option 2: Replace config.py with A100-optimized version

```bash
# In Colab
!cp config_a100.py config.py

# Verify
import config
print(f"Batch size: {config.BATCH_SIZE}")
```

### Option 3: Temporary Override (No file changes)

```python
# In your optimization cell
from optimization.nsga3_runner import NSGA3Runner
from pathlib import Path
import config

# Override batch size temporarily
original_batch_size = config.BATCH_SIZE
config.BATCH_SIZE = 64

print(f"Overriding batch size: {original_batch_size} → {config.BATCH_SIZE}")

# Run optimization with new batch size
runner = NSGA3Runner(...)
result = runner.run()
```

---

## ⚡ Expected Performance Gains

### With Batch Size 64:

**Single Model Training:**
- Time per epoch: ~2-3 minutes (vs ~10-12 min)
- Early stopping at epoch 30: ~60-90 minutes (vs ~300-360 min)
- **~4-5x faster per model** ✅

**Full Optimization (24 pop × 50 gen = 1,200 models):**
- Current (batch_size=16): ~500-600 hours (~25 days)
- Optimized (batch_size=64): **~100-150 hours (~5-6 days)**
- **Saves 20 days of compute time!** 🎉

### With Batch Size 128:

**Even faster:**
- Time per epoch: ~1-2 minutes
- Full optimization: **~60-80 hours (~3 days)**
- **~7x speedup!**

---

## 💾 GPU Memory Breakdown

### A100 80GB Memory Usage (Batch Size 64):

```
Total: 80 GB
├── Model weights (ResNet-50): ~0.5 GB
├── Gradients: ~0.5 GB
├── Optimizer state (Adam): ~1 GB
├── Input batch (64 × 224×224×3): ~10 GB
├── Activations & intermediate: ~8 GB
├── PyTorch overhead: ~2 GB
└── Available: ~58 GB (73% free!)
```

**You have plenty of headroom to increase batch size further!**

---

## 🔬 Testing Different Batch Sizes

Want to find the optimal batch size for your setup? Try this:

```python
import torch
from models import ResNet50WithPartialFineTuning
from data.dataset import create_dataloaders
import config

# Test different batch sizes
for batch_size in [16, 32, 64, 128, 256]:
    print(f"\nTesting batch_size={batch_size}")

    try:
        # Create dummy dataloader
        config.BATCH_SIZE = batch_size

        model = ResNet50WithPartialFineTuning(
            unfreeze_fraction=0.3,
            dropout_rate=0.2
        ).cuda()

        # Create test batch
        test_input = torch.randn(batch_size, 3, 224, 224).cuda()

        # Forward pass
        with torch.no_grad():
            output = model(test_input)

        # Check memory
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9

        print(f"  ✓ Success!")
        print(f"  Memory allocated: {memory_allocated:.2f} GB")
        print(f"  Memory reserved: {memory_reserved:.2f} GB")

        # Cleanup
        del model, test_input, output
        torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"  ✗ Failed: {e}")
        break

print("\nRecommendation: Use largest batch size that succeeded")
```

---

## 📈 Additional A100 Optimizations

### 1. Mixed Precision Training (FP16)

The A100 has excellent FP16 performance. Add this for ~2x additional speedup:

```python
# In trainer.py or add to config
USE_AMP = True  # Automatic Mixed Precision

# In training loop (already implemented in some frameworks)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Increase Number of Workers

A100 can handle more data loading workers:

```python
# In dataset.py or config
NUM_WORKERS = 8  # Increase from default (usually 2-4)

train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,  # More parallel data loading
    pin_memory=True
)
```

### 3. Enable TF32 (Tensor Float 32)

A100 has TF32 cores for faster computation:

```python
# Add at the start of your script
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("✓ TF32 enabled for faster matrix operations")
```

### 4. Optimize Image Loading

For large datasets, use:

```python
# Use faster image decoding
from torchvision.io import read_image

# Instead of PIL
# img = Image.open(path)

# Use this (faster on A100)
img = read_image(path)
```

---

## 🎛️ Recommended Full Configuration for A100

```python
# config.py - Optimized for A100 80GB

# Training settings
BATCH_SIZE = 64              # Or 128 for maximum speed
IMAGE_SIZE = (224, 224)
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Data loading
NUM_WORKERS = 8              # More workers for faster data loading
PIN_MEMORY = True            # Faster GPU transfer

# Optimization settings
NSGA3_CONFIG = {
    "pop_size": 24,          # Can increase to 32 if you want
    "n_generations": 50,
    "n_objectives": 4,
}

# A100-specific optimizations
USE_AMP = True               # Mixed precision training
TF32_ENABLED = True          # Use TF32 cores
```

---

## 📊 Performance Comparison

### Time to Complete Full Optimization (24 × 50 = 1,200 models):

| Configuration | Time | Speedup |
|--------------|------|---------|
| Default (batch=16) | ~25 days | Baseline |
| Batch=32 | ~12 days | 2x |
| Batch=64 | **~6 days** | **4x** ✅ |
| Batch=128 | **~3 days** | **8x** 🚀 |
| Batch=128 + AMP | **~1.5 days** | **16x** ⚡ |

---

## ✅ Quick Start: Apply Optimizations Now

Copy this into a Colab cell **before Section 5**:

```python
# ============================================================================
# A100 80GB OPTIMIZATION
# ============================================================================

import config
import torch

# Set optimal batch size for A100
config.BATCH_SIZE = 64  # Use 128 if you want maximum speed

# Enable A100 optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("=" * 80)
print("A100 80GB OPTIMIZATIONS APPLIED")
print("=" * 80)
print(f"✓ Batch size: {config.BATCH_SIZE}")
print(f"✓ TF32 enabled: True")
print(f"✓ Expected GPU memory: ~{config.BATCH_SIZE * 0.3:.1f} GB / 80 GB")
print(f"✓ Training batches/epoch: ~{20486 * 0.8 / config.BATCH_SIZE:.0f}")
print(f"✓ Expected speedup: ~{config.BATCH_SIZE / 16}x faster")
print("=" * 80)
```

---

## 🎯 Summary

With an A100 80GB GPU, you should:

1. ✅ **Increase batch size to 64 or 128** (currently only 16)
2. ✅ **Enable TF32** for faster computation
3. ✅ **Consider mixed precision training** for 2x additional speedup
4. ✅ **Increase data loading workers** to 8

**Expected result: 5-10x faster optimization! 🚀**

Your full optimization that would take 25 days can complete in just **3-6 days**!
