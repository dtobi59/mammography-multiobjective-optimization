import json

# Read the notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# The optimization code cell to insert (after cell 20 - the "5a" markdown header)
optimization_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-opt",
    "metadata": {},
    "outputs": [],
    "source": [
"""# ============================================================================
# OPTIMIZED FOR A100 80GB GPU
# ============================================================================

import torch
import config
from optimization.nsga3_runner import NSGA3Runner
from pathlib import Path

# Apply A100 GPU optimizations
print("=" * 80)
print("A100 80GB GPU OPTIMIZATIONS")
print("=" * 80)

# Enable TF32 for faster matrix operations on A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("✓ TF32 enabled for tensor cores")

# Verify batch size is optimized
print(f"✓ Batch size: {config.BATCH_SIZE}")
print(f"✓ Expected GPU memory: ~{config.BATCH_SIZE * 0.3:.1f} GB / 80 GB")
print(f"✓ Training batches/epoch: ~{20486 * 0.8 / config.BATCH_SIZE:.0f}")
print(f"✓ Speedup vs batch_size=16: ~{config.BATCH_SIZE / 16:.1f}x faster")

# Optional: Scale learning rate with batch size
if config.BATCH_SIZE != 16:
    lr_scale = (config.BATCH_SIZE / 16) ** 0.5  # sqrt scaling
    old_lr = config.HYPERPARAMETER_BOUNDS["learning_rate"]
    config.HYPERPARAMETER_BOUNDS["learning_rate"] = (
        old_lr[0] * lr_scale,
        old_lr[1] * lr_scale
    )
    print(f"✓ Learning rate scaled by {lr_scale:.3f}x for larger batches")
    print(f"  New range: ({config.HYPERPARAMETER_BOUNDS['learning_rate'][0]:.2e}, "
          f"{config.HYPERPARAMETER_BOUNDS['learning_rate'][1]:.2e})")

print("=" * 80)
print()

# Split data into train and validation
from sklearn.model_selection import train_test_split

# Patient-wise split (prevent data leakage)
unique_patients = vindr_metadata['patient_id'].unique()
train_patients, val_patients = train_test_split(
    unique_patients,
    train_size=config.TRAIN_VAL_SPLIT,
    random_state=config.RANDOM_SEED
)

train_metadata = vindr_metadata[vindr_metadata['patient_id'].isin(train_patients)]
val_metadata = vindr_metadata[vindr_metadata['patient_id'].isin(val_patients)]

print("=" * 80)
print("DATA SPLIT")
print("=" * 80)
print(f"Training:   {len(train_metadata)} images from {len(train_patients)} patients")
print(f"Validation: {len(val_metadata)} images from {len(val_patients)} patients")
print(f"Split ratio: {config.TRAIN_VAL_SPLIT:.0%} train / {1-config.TRAIN_VAL_SPLIT:.0%} val")
print("=" * 80)
print()

# Setup optimizer
image_dir = str(Path(config.VINDR_MAMMO_PATH) / config.VINDR_CONFIG["image_dir"])

print("=" * 80)
print("STARTING OPTIMIZATION")
print("=" * 80)
print(f"Population size: {config.NSGA3_CONFIG['pop_size']}")
print(f"Generations: {config.NSGA3_CONFIG['n_generations']}")
print(f"Total models to train: {config.NSGA3_CONFIG['pop_size'] * config.NSGA3_CONFIG['n_generations']}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
print(f"Results: {OUTPUT_DIR}")
print()
print(f"Estimated time per model: ~7 minutes")
print(f"Estimated time per generation: ~{24 * 7 / 60:.1f} hours")
print(f"Estimated total time: ~{24 * 50 * 7 / 60 / 24:.1f} days")
print("=" * 80)
print()

# Create runner
runner = NSGA3Runner(
    train_metadata=train_metadata,
    val_metadata=val_metadata,
    image_dir=image_dir,
    output_dir=OUTPUT_DIR,          # Google Drive
    checkpoint_dir=CHECKPOINT_DIR,  # Google Drive
    save_frequency=1                # Save after every generation
)

# Run optimization
print("🚀 Starting optimization...")
print("All progress will be saved to Google Drive automatically!")
print()

result = runner.run()

print()
print("=" * 80)
print("OPTIMIZATION COMPLETE!")
print("=" * 80)
print(f"Pareto front size: {len(result.F)}")
print(f"Results saved to: {runner.output_dir}")
print("=" * 80)
"""
    ]
}

# Insert the optimization cell after cell 20 (the "5a" markdown header)
notebook['cells'].insert(21, optimization_cell)

# Write the updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("[OK] Added optimization cell with A100 optimizations to colab_tutorial.ipynb")
print("     Location: Cell 21 (after '5a. Start New Optimization' header)")
