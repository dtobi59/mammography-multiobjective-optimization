# Checkpoint Storage Locations

## Overview

The optimization process stores checkpoints in **two separate directories**:

1. **Model Checkpoints** - Trained model weights for each evaluation
2. **Optimization Checkpoints** - NSGA-III algorithm state and Pareto fronts

---

## 1. Model Checkpoints

**Default Location:** `./checkpoints/`

**Purpose:** Store individual trained model weights during each hyperparameter evaluation

**Structure:**
```
./checkpoints/
├── eval_1/
│   └── best_checkpoint.pt
├── eval_2/
│   └── best_checkpoint.pt
├── eval_3/
│   └── best_checkpoint.pt
└── ...
```

**Configuration:**
```python
runner = NSGA3Runner(
    train_metadata=train_metadata,
    val_metadata=val_metadata,
    image_dir=image_dir,
    checkpoint_dir="./checkpoints",  # Model checkpoints go here
    ...
)
```

**In Google Colab:**
- Location: `/content/mammography-multiobjective-optimization/checkpoints/`
- Each `eval_X` folder contains the best model from that hyperparameter configuration

---

## 2. Optimization Checkpoints

**Default Location:** `./optimization_results/optimization_checkpoints/`

**Purpose:** Store NSGA-III algorithm state, population, and Pareto front at each generation

**Structure:**
```
./optimization_results/
├── optimization_checkpoints/
│   ├── checkpoint_gen_0001.pkl    # Generation 1 state
│   ├── checkpoint_gen_0002.pkl    # Generation 2 state
│   ├── checkpoint_gen_0003.pkl    # Generation 3 state
│   ├── ...
│   ├── pareto_gen_0001.csv        # Pareto front at gen 1
│   ├── pareto_gen_0002.csv        # Pareto front at gen 2
│   └── pareto_gen_0003.csv        # Pareto front at gen 3
└── pareto_solutions_TIMESTAMP.csv  # Final Pareto front
```

**Configuration:**
```python
runner = NSGA3Runner(
    train_metadata=train_metadata,
    val_metadata=val_metadata,
    image_dir=image_dir,
    output_dir="./optimization_results",  # Optimization checkpoints go in {output_dir}/optimization_checkpoints/
    save_frequency=1,  # Save every generation
    ...
)
```

**In Google Colab:**
- Location: `/content/mammography-multiobjective-optimization/optimization_results/optimization_checkpoints/`

---

## Checkpoint Contents

### Model Checkpoint (`best_checkpoint.pt`)
```python
{
    'model_state_dict': ...,      # Trained model weights
    'optimizer_state_dict': ...,  # Optimizer state
    'epoch': ...,                 # Best epoch number
    'metrics': {                  # Validation metrics
        'pr_auc': ...,
        'auroc': ...,
        'brier': ...,
    }
}
```

### Optimization Checkpoint (`checkpoint_gen_XXXX.pkl`)
```python
{
    'generation': ...,            # Generation number
    'algorithm': ...,             # NSGA-III algorithm state
    'population': ...,            # Current population
    'objectives': ...,            # Objective values
    'hyperparameters': ...,       # Hyperparameter configurations
}
```

### Pareto Front CSV (`pareto_gen_XXXX.csv`)
```csv
solution_id,learning_rate,weight_decay,dropout_rate,augmentation_strength,unfreeze_fraction,pr_auc,auroc,brier,robustness_degradation
0,0.001234,0.000567,...
1,0.002345,0.000678,...
...
```

---

## Using Checkpoints

### List Available Checkpoints
```python
from optimization.nsga3_runner import NSGA3Runner

# After creating runner
runner = NSGA3Runner(...)

# List all checkpoints
checkpoints = runner.list_checkpoints()
print(f"Found {len(checkpoints)} checkpoints")
for checkpoint in checkpoints:
    print(checkpoint)
```

### Resume from Checkpoint
```python
# Load a specific checkpoint
checkpoint_data = runner.load_checkpoint(checkpoints[-1])  # Latest checkpoint

# Get Pareto front from checkpoint
pareto_df = runner.get_pareto_front_from_checkpoint(checkpoints[-1])
print(pareto_df)
```

### Load Trained Model
```python
import torch
from models import ResNet50WithPartialFineTuning

# Load a specific model
model = ResNet50WithPartialFineTuning(
    unfreeze_fraction=0.3,
    dropout_rate=0.2
)

checkpoint_path = "./checkpoints/eval_42/best_checkpoint.pt"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation PR-AUC: {checkpoint['metrics']['pr_auc']:.4f}")
```

---

## Saving Checkpoints to Google Drive

To persist checkpoints across Colab sessions:

```python
import shutil

# After optimization completes
# Copy checkpoints to Google Drive
shutil.copytree(
    "./checkpoints",
    "/content/drive/MyDrive/vindr_optimization/checkpoints"
)

shutil.copytree(
    "./optimization_results",
    "/content/drive/MyDrive/vindr_optimization/optimization_results"
)

print("Checkpoints saved to Google Drive!")
```

---

## Storage Requirements

### Approximate Sizes:
- **Model checkpoint**: ~90 MB per evaluation (ResNet-50 weights)
- **Optimization checkpoint**: ~5-10 MB per generation
- **Pareto CSV**: ~100 KB per generation

### Full Run Estimates:
For default configuration (pop_size=24, n_generations=50):
- Model checkpoints: ~24 × 50 × 90 MB = **108 GB**
- Optimization checkpoints: ~50 × 10 MB = **500 MB**
- **Total**: ~110 GB

**Recommendation:**
- For Colab: Save final results to Google Drive regularly
- For local runs: Ensure sufficient disk space
- Consider reducing `save_frequency` to save every N generations

---

## Cleanup

To save space, you can delete intermediate checkpoints:

```python
import glob
import os

# Keep only every 5th generation
checkpoints = sorted(glob.glob("./optimization_results/optimization_checkpoints/checkpoint_gen_*.pkl"))
for i, checkpoint in enumerate(checkpoints):
    gen = int(checkpoint.split("_")[-1].replace(".pkl", ""))
    if gen % 5 != 0:  # Keep every 5th generation
        os.remove(checkpoint)
        print(f"Deleted {checkpoint}")
```
