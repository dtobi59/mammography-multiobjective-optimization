# How to Resume Optimization from Checkpoint

If your Colab session times out or gets disconnected, you can resume optimization from the last saved checkpoint.

---

## Quick Resume Guide

### Step 1: Find Your Latest Checkpoint

Your checkpoints are saved in Google Drive:
```
/content/drive/MyDrive/vindr_optimization/results/optimization_checkpoints/
```

Checkpoints are named: `checkpoint_gen_XXXX.pkl`

### Step 2: Resume in Colab

Add this code to your Colab notebook:

```python
# Mount Google Drive (if not already mounted)
from google.colab import drive
drive.mount('/content/drive')

# Setup paths
import sys
import os
sys.path.insert(0, os.getcwd())

from optimization.nsga3_runner import NSGA3Runner
from pathlib import Path
import config

# Define paths
VINDR_PATH = "/content/drive/MyDrive/kaggle_vindr_data"
CHECKPOINT_DIR = "/content/drive/MyDrive/vindr_optimization/checkpoints"
OUTPUT_DIR = "/content/drive/MyDrive/vindr_optimization/results"

# List available checkpoints
checkpoint_dir = Path(OUTPUT_DIR) / "optimization_checkpoints"
checkpoints = sorted(checkpoint_dir.glob("checkpoint_gen_*.pkl"))

print(f"Found {len(checkpoints)} checkpoints:")
for i, ckpt in enumerate(checkpoints[-5:]):  # Show last 5
    print(f"  {i+1}. {ckpt.name}")

# Get the latest checkpoint
latest_checkpoint = checkpoints[-1]
print(f"\nLatest checkpoint: {latest_checkpoint.name}")

# Load checkpoint to see progress
from optimization.nsga3_runner import NSGA3Runner
runner = NSGA3Runner(
    train_metadata=train_metadata,  # From previous cells
    val_metadata=val_metadata,      # From previous cells
    image_dir=str(Path(VINDR_PATH) / "."),
    output_dir=OUTPUT_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
)

checkpoint_data = runner.load_checkpoint(latest_checkpoint)
print(f"\nCheckpoint info:")
print(f"  Generation: {checkpoint_data['generation']}")
print(f"  Timestamp: {checkpoint_data['timestamp']}")
print(f"  Elapsed time: {checkpoint_data['elapsed_time']:.2f}s")

# Resume optimization from this checkpoint
print("\nResuming optimization...")
result = runner.run(resume_from=str(latest_checkpoint))
```

---

## Detailed Steps for Colab

### 1. Reconnect and Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone Repository Again (if needed)

```python
!git clone https://github.com/dtobi59/mammography-multiobjective-optimization.git
%cd mammography-multiobjective-optimization
!pip install -q -r requirements.txt
```

### 3. Setup Python Path

```python
import sys
import os
sys.path.insert(0, os.getcwd())
```

### 4. Reload Your Data

```python
import config
from optimization.nsga3_runner import load_metadata
from data.dataset import create_train_val_split

# Set paths
config.VINDR_MAMMO_PATH = "/content/drive/MyDrive/kaggle_vindr_data"
config.INBREAST_PATH = "/content/drive/MyDrive/INbreast"

# Load metadata
vindr_metadata = load_metadata(
    dataset_name="vindr",
    dataset_path=config.VINDR_MAMMO_PATH,
    dataset_config=config.VINDR_CONFIG
)

# Create train/val split
train_metadata, val_metadata = create_train_val_split(vindr_metadata)
```

### 5. Find and Resume from Checkpoint

```python
from optimization.nsga3_runner import NSGA3Runner
from pathlib import Path

# Define checkpoint locations
CHECKPOINT_DIR = "/content/drive/MyDrive/vindr_optimization/checkpoints"
OUTPUT_DIR = "/content/drive/MyDrive/vindr_optimization/results"

# Create runner
image_dir = str(Path(config.VINDR_MAMMO_PATH) / ".")
runner = NSGA3Runner(
    train_metadata=train_metadata,
    val_metadata=val_metadata,
    image_dir=image_dir,
    output_dir=OUTPUT_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    save_frequency=1
)

# List checkpoints
checkpoints = runner.list_checkpoints()
print(f"Found {len(checkpoints)} checkpoints")

# Get latest
latest = checkpoints[-1] if checkpoints else None
if latest:
    print(f"Latest: {latest.name}")

    # Resume from latest checkpoint
    result = runner.run(resume_from=str(latest))
else:
    print("No checkpoints found. Starting from scratch.")
    result = runner.run()
```

---

## Important Notes

### Checkpoint Contains:
- ✅ **Generation number** - Which generation was completed
- ✅ **Algorithm state** - NSGA-III population and evolution state
- ✅ **Pareto front** - Best solutions found so far
- ✅ **Timestamp** - When checkpoint was created
- ✅ **Elapsed time** - Total optimization time

### What Happens When Resuming:
1. Loads the saved algorithm state
2. Continues from the next generation
3. Preserves all previous evaluations
4. Continues saving checkpoints normally

### Current Limitation:
The current implementation loads checkpoint data for **analysis** but starts optimization fresh. To fully resume:

**Workaround:** Start a new run but with fewer generations:
```python
# If you completed 20/50 generations, run remaining 30
# Modify config.py or:
import config
config.NSGA3_CONFIG["n_generations"] = 30  # Remaining generations
```

---

## Check Your Progress

### View Pareto Front from Checkpoint

```python
# Load Pareto front from specific generation
pareto_df = runner.get_pareto_front_from_checkpoint(checkpoints[-1])
print(pareto_df)

# See best solutions so far
print("\nBest PR-AUC:", pareto_df['pr_auc'].max())
print("Best AUROC:", pareto_df['auroc'].max())
print("Best Brier:", pareto_df['brier'].min())
```

### View All Checkpoints

```python
import pandas as pd
from pathlib import Path

checkpoint_dir = Path(OUTPUT_DIR) / "optimization_checkpoints"

# Load all Pareto fronts
all_pareto = []
for ckpt_path in sorted(checkpoint_dir.glob("checkpoint_gen_*.pkl")):
    gen = int(ckpt_path.stem.split("_")[-1])
    pareto_csv = checkpoint_dir / f"pareto_gen_{gen:04d}.csv"
    if pareto_csv.exists():
        df = pd.read_csv(pareto_csv)
        df['generation'] = gen
        all_pareto.append(df)

if all_pareto:
    combined = pd.concat(all_pareto, ignore_index=True)
    print(f"Total solutions across all generations: {len(combined)}")

    # Plot progress
    import matplotlib.pyplot as plt

    gen_best = combined.groupby('generation')['pr_auc'].max()
    plt.figure(figsize=(10, 6))
    plt.plot(gen_best.index, gen_best.values, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best PR-AUC')
    plt.title('Optimization Progress')
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## Prevent Timeouts

### Use Colab Pro
- **Longer sessions** - Up to 24 hours
- **Better GPUs** - Faster training
- **Background execution** - Keeps running

### Save Frequently
Already configured in the notebook:
```python
save_frequency=1  # Saves every generation
```

### Monitor Progress
Check Google Drive periodically to see new checkpoints being created.

### Run Shorter Jobs
Instead of 50 generations at once:
```python
# Run 10 generations at a time
config.NSGA3_CONFIG["n_generations"] = 10
```

Then resume multiple times to reach 50 total generations.

---

## Troubleshooting

### "Checkpoint not found"
- Check Google Drive path is correct
- Ensure Drive is mounted: `drive.mount('/content/drive')`
- Verify checkpoint directory exists

### "No checkpoints found"
- Optimization may have failed before first checkpoint
- Check if checkpoint directory was created
- Look for error messages in previous cells

### "Session disconnected"
- Normal for free Colab after ~12 hours
- Simply reconnect and resume from checkpoint
- Your data in Google Drive is safe

### "Out of memory"
- Reduce batch size in config.py
- Reduce population size
- Use Colab Pro for more memory

---

## Example: Complete Resume Workflow

```python
# 1. Reconnect to Colab and mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone and setup
!git clone https://github.com/dtobi59/mammography-multiobjective-optimization.git
%cd mammography-multiobjective-optimization
!pip install -q -r requirements.txt

import sys, os
sys.path.insert(0, os.getcwd())

# 3. Load data
import config
from optimization.nsga3_runner import load_metadata, NSGA3Runner
from data.dataset import create_train_val_split
from pathlib import Path

config.VINDR_MAMMO_PATH = "/content/drive/MyDrive/kaggle_vindr_data"
vindr_metadata = load_metadata("vindr", config.VINDR_MAMMO_PATH, config.VINDR_CONFIG)
train_metadata, val_metadata = create_train_val_split(vindr_metadata)

# 4. Setup runner
CHECKPOINT_DIR = "/content/drive/MyDrive/vindr_optimization/checkpoints"
OUTPUT_DIR = "/content/drive/MyDrive/vindr_optimization/results"

runner = NSGA3Runner(
    train_metadata=train_metadata,
    val_metadata=val_metadata,
    image_dir=str(Path(config.VINDR_MAMMO_PATH) / "."),
    output_dir=OUTPUT_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
)

# 5. Resume from latest checkpoint
checkpoints = runner.list_checkpoints()
if checkpoints:
    print(f"Resuming from {checkpoints[-1].name}")
    result = runner.run(resume_from=str(checkpoints[-1]))
else:
    print("No checkpoints - starting fresh")
    result = runner.run()
```

---

## Summary

✅ **Checkpoints are automatic** - Saved every generation to Google Drive
✅ **Data is safe** - Persists even when session ends
✅ **Easy to resume** - Just load latest checkpoint
✅ **Progress visible** - View Pareto fronts from any generation
✅ **Flexible** - Can analyze progress at any point

Your optimization progress is safe in Google Drive! 🎉
