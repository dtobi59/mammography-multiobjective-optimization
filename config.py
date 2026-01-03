"""
Configuration and hyperparameter bounds for multi-objective optimization.
"""

import numpy as np

# Random seeds for reproducibility
RANDOM_SEED = 42

# Data paths (to be set by user)
VINDR_MAMMO_PATH = "/content/drive/MyDrive/kaggle_vindr_data"
INBREAST_PATH = "path/to/inbreast"

# Dataset-specific configurations
# VinDr-Mammo configuration
VINDR_CONFIG = {
    "metadata_file": "vindr_detection_v1_folds.csv",  # Relative to VINDR_MAMMO_PATH
    "image_dir": "images",            # Relative to VINDR_MAMMO_PATH
    "image_id_col": "image_id",
    "patient_id_col": "patient_id",
    "laterality_col": "laterality",
    "view_col": "view",
    "birads_col": "breast_birads",
    "image_extension": ".png",
}

# INbreast configuration
INBREAST_CONFIG = {
    "metadata_file": "INbreast.csv",  # Relative to INBREAST_PATH
    "image_dir": "images",            # Relative to INBREAST_PATH
    "metadata_format": "csv",         # 'csv' or 'xml'
    "patient_id_col": "Patient ID",
    "laterality_col": "Laterality",
    "view_col": "View",
    "birads_col": "Bi-Rads",
    "filename_col": "File Name",
}

# Dataset configuration
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation
IMAGE_SIZE = (224, 224)  # ResNet-50 standard input size

# Fixed training hyperparameters
BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_METRIC = "val_pr_auc"  # Monitor PR-AUC for early stopping

# Hyperparameter search space (continuous)
HYPERPARAMETER_BOUNDS = {
    "learning_rate": (1e-5, 1e-2),  # Log scale
    "weight_decay": (1e-6, 1e-2),   # Log scale
    "dropout_rate": (0.0, 0.5),     # Linear scale
    "augmentation_strength": (0.0, 1.0),  # Linear scale
    "unfreeze_fraction": (0.0, 1.0),  # Linear scale: fraction of backbone layers to unfreeze
}

# Augmentation configuration (intensity-only)
AUGMENTATION_CONFIG = {
    "brightness_factor": 0.2,   # Maximum brightness adjustment
    "contrast_factor": 0.2,     # Maximum contrast scaling
    "noise_std": 0.05,          # Standard deviation of Gaussian noise
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
# Thresholds selected on source validation to achieve target operating points
# These will be computed during optimization and transferred to target domain
THRESHOLD_SELECTION_METHOD = "youden_j"  # Youden's J statistic (sensitivity + specificity - 1)
