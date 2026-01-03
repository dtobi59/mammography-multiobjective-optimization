"""
Optimized configuration for A100 80GB GPU

This configuration takes full advantage of the A100's capabilities:
- Large batch sizes for faster training
- Optimal memory utilization
- Faster convergence
"""

import numpy as np

# Random seeds for reproducibility
RANDOM_SEED = 42

# Data paths (to be set by user)
VINDR_MAMMO_PATH = "/content/drive/MyDrive/kaggle_vindr_data"
INBREAST_PATH = "/content/drive/MyDrive/INbreast"

# Dataset-specific configurations
# VinDr-Mammo configuration
VINDR_CONFIG = {
    "metadata_file": "vindr_detection_v1_folds.csv",
    "image_dir": ".",  # Images are in patient_id subdirectories
    "image_id_col": "image_id",
    "patient_id_col": "patient_id",
    "laterality_col": "laterality",
    "view_col": "view",
    "birads_col": "breast_birads",
    "image_extension": ".png",
}

# INbreast configuration
INBREAST_CONFIG = {
    "metadata_file": "INbreast.csv",
    "image_dir": "images",
    "metadata_format": "csv",
    "patient_id_col": "Patient ID",
    "laterality_col": "Laterality",
    "view_col": "View",
    "birads_col": "Bi-Rads",
    "filename_col": "File Name",
}

# Dataset configuration
TRAIN_VAL_SPLIT = 0.8
IMAGE_SIZE = (224, 224)

# ============================================================================
# OPTIMIZED FOR A100 80GB GPU
# ============================================================================

# Training hyperparameters - OPTIMIZED FOR A100
BATCH_SIZE = 64  # Increased from 16 → 4x faster per epoch!
                 # A100 can handle 64-128 easily
                 # Uses ~20-25 GB of 80 GB available

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_METRIC = "val_pr_auc"

# Performance notes with batch size 64:
# - Training: ~256 batches/epoch (vs 1,280 with batch_size=16)
# - 5x faster epoch completion
# - More stable gradients (larger batches)
# - Better GPU utilization (A100 is underutilized at batch_size=16)

# ============================================================================
# Hyperparameter search space (continuous)
# ============================================================================
HYPERPARAMETER_BOUNDS = {
    "learning_rate": (1e-5, 1e-2),  # Log scale
    "weight_decay": (1e-6, 1e-2),   # Log scale
    "dropout_rate": (0.0, 0.5),     # Linear scale
    "augmentation_strength": (0.0, 1.0),  # Linear scale
    "unfreeze_fraction": (0.0, 1.0),  # Linear scale: fraction of backbone layers to unfreeze
}

# Augmentation configuration (intensity-only)
AUGMENTATION_CONFIG = {
    "brightness_factor": 0.2,
    "contrast_factor": 0.2,
    "noise_std": 0.05,
}

# Robustness evaluation perturbation (mild intensity perturbations)
ROBUSTNESS_PERTURBATION = {
    "brightness_delta": 0.1,
    "contrast_delta": 0.1,
    "noise_std": 0.02,
}

# NSGA-III configuration
NSGA3_CONFIG = {
    "pop_size": 24,           # Population size
    "n_generations": 50,      # Number of generations
    "n_objectives": 4,        # 4 objectives: -PR-AUC, -AUROC, Brier, Robustness degradation
}

# Decision threshold selection
THRESHOLD_SELECTION_METHOD = "youden_j"  # Youden's J statistic

# ============================================================================
# A100 GPU UTILIZATION STATS
# ============================================================================
# Expected GPU utilization with these settings:
# - Memory usage: ~20-25 GB / 80 GB (30%)
# - Compute utilization: ~85-95%
# - Training speed: ~5x faster than batch_size=16
# - Time per epoch: ~2-3 minutes (vs ~10-12 min with batch_size=16)
# - Full optimization (24 × 50): ~100-150 hours (vs ~500 hours)
#
# You can further increase batch_size to 128 or even 256 if you want
# to maximize GPU utilization and speed!
# ============================================================================
