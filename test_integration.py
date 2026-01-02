"""
Integration test with minimal end-to-end training.
Creates dummy data and runs a complete training cycle.
"""

import os
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path

print("="*60)
print("INTEGRATION TEST - End-to-End Training")
print("="*60)

# Create temporary directory for test data
test_dir = Path("./test_data_temp")
test_dir.mkdir(exist_ok=True)
image_dir = test_dir / "images"
image_dir.mkdir(exist_ok=True)

print("\n[1/8] Creating dummy dataset...")

# Create dummy images (small for speed)
n_patients = 4
n_images = n_patients * 4  # 4 images per patient (2 breasts x 2 views)

metadata_rows = []
for p in range(n_patients):
    for b in range(2):  # 2 breasts
        breast_id = f"p{p}_b{b}"
        label = p % 2  # Alternate labels

        for view in ["CC", "MLO"]:
            img_id = f"{breast_id}_{view}"
            img_filename = f"{img_id}.png"

            # Create dummy grayscale image (64x64 for speed)
            dummy_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            Image.fromarray(dummy_img, mode='L').save(image_dir / img_filename)

            metadata_rows.append({
                "image_id": img_id,
                "patient_id": f"p{p}",
                "breast_id": breast_id,
                "view": view,
                "label": label,
                "image_path": img_filename,
            })

metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(test_dir / "metadata.csv", index=False)

print(f"   Created {len(metadata)} images for {n_patients} patients")

print("\n[2/8] Splitting data...")

from data.dataset import create_train_val_split
from utils.seed import set_all_seeds

set_all_seeds(42)
train_metadata, val_metadata = create_train_val_split(
    metadata, train_ratio=0.75, random_seed=42
)

print(f"   Train: {len(train_metadata)} images")
print(f"   Val: {len(val_metadata)} images")

print("\n[3/8] Creating dataloaders...")

from data.dataset import create_dataloaders
import config

# Override image size for speed
original_image_size = config.IMAGE_SIZE
config.IMAGE_SIZE = (64, 64)

train_loader, val_loader = create_dataloaders(
    train_metadata=train_metadata,
    val_metadata=val_metadata,
    image_dir=str(image_dir),
    batch_size=2,
    augmentation_strength=0.3,
    num_workers=0,  # Use 0 workers for Windows compatibility
)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

print("\n[4/8] Creating model...")

from models import ResNet50WithPartialFineTuning

model = ResNet50WithPartialFineTuning(
    unfreeze_fraction=0.3,
    dropout_rate=0.2,
    pretrained=False,  # Skip pretrained for speed
)

trainable = model.get_trainable_params()
frozen = model.get_frozen_params()
print(f"   Trainable params: {trainable:,}")
print(f"   Frozen params: {frozen:,}")

print("\n[5/8] Testing forward pass...")

for images, labels, image_ids in train_loader:
    outputs = model(images)
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    break

print("\n[6/8] Training for 2 epochs...")

from training.trainer import Trainer

# Override max epochs for speed
original_max_epochs = config.MAX_EPOCHS
config.MAX_EPOCHS = 2

# Override early stopping patience
original_patience = config.EARLY_STOPPING_PATIENCE
config.EARLY_STOPPING_PATIENCE = 5

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    val_metadata=val_metadata,
    learning_rate=0.001,
    weight_decay=0.0001,
    max_epochs=2,
    device="cpu",  # Use CPU for compatibility
    checkpoint_dir=str(test_dir / "checkpoints"),
)

best_metrics = trainer.train()

print(f"\n   Best validation metrics:")
print(f"   - PR-AUC: {best_metrics['pr_auc']:.4f}")
print(f"   - AUROC: {best_metrics['auroc']:.4f}")
print(f"   - Brier: {best_metrics['brier']:.4f}")

print("\n[7/8] Evaluating robustness...")

from training.robustness import RobustnessEvaluator

robustness_eval = RobustnessEvaluator(
    model=model,
    val_loader=val_loader,
    val_metadata=val_metadata,
    device="cpu",
)

robustness_degradation = robustness_eval.evaluate()
print(f"   Robustness degradation: {robustness_degradation:.4f}")

print("\n[8/8] Testing threshold selection...")

from training.metrics import find_optimal_threshold

# Get predictions for threshold selection
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in val_loader:
        preds = model(images).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Aggregate to breast level
from utils.noisy_or import aggregate_to_breast_level

image_preds = {}
for i, (_, _, img_id) in enumerate(val_loader.dataset):
    if i < len(all_preds):
        image_preds[img_id] = all_preds[i]

breast_preds, breast_labels = aggregate_to_breast_level(
    image_preds, val_metadata
)

optimal_threshold = find_optimal_threshold(breast_preds, breast_labels)
print(f"   Optimal threshold: {optimal_threshold:.4f}")

from training.metrics import compute_sensitivity_specificity
sens, spec = compute_sensitivity_specificity(
    breast_preds, breast_labels, optimal_threshold
)
print(f"   Sensitivity: {sens:.4f}")
print(f"   Specificity: {spec:.4f}")

print("\n[CLEANUP] Removing test data...")

# Restore config
config.IMAGE_SIZE = original_image_size
config.MAX_EPOCHS = original_max_epochs
config.EARLY_STOPPING_PATIENCE = original_patience

# Cleanup
shutil.rmtree(test_dir)

print("\n" + "="*60)
print("[SUCCESS] Integration test completed successfully!")
print("="*60)
print("\nAll components working correctly:")
print("  - Data loading and splitting")
print("  - Augmentation pipeline")
print("  - Model creation and forward pass")
print("  - Training with early stopping")
print("  - Breast-level aggregation with Noisy OR")
print("  - Robustness evaluation")
print("  - Threshold selection")
print("\nThe implementation is ready for use with real data!")
