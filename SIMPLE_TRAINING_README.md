# Simple Mammography Training & Inference

Simplified training and inference notebooks using ResNet-50 backbone for breast cancer classification.

## Overview

These notebooks provide a straightforward alternative to the full multi-objective optimization pipeline. They're ideal for:
- Quick baseline experiments
- Understanding the core model architecture
- Testing different hyperparameter configurations
- Educational purposes

---

## Notebooks

### 1. Simple_Training_VinDr.ipynb

**Purpose**: Train a ResNet-50 model on preprocessed VinDr-Mammo dataset.

**Features**:
- ResNet-50 pretrained on ImageNet
- Partial fine-tuning (configurable fraction of layers)
- Dropout regularization
- Binary classification with sigmoid output
- Patient-level train/val split (prevents data leakage)
- Breast-level evaluation using Noisy-OR aggregation
- Early stopping based on validation PR-AUC
- Intensity-based data augmentation
- Training history visualization

**Architecture**:
```
Input (224x224x3)
    ↓
ResNet-50 Backbone (2048 features)
    ↓
Dropout (configurable rate)
    ↓
Linear Classifier (2048 → 1)
    ↓
Sigmoid
    ↓
Output (probability)
```

**Configurable Hyperparameters**:
- `LEARNING_RATE`: 1e-4 (default)
- `WEIGHT_DECAY`: 1e-4 (default)
- `DROPOUT_RATE`: 0.3 (default)
- `UNFREEZE_FRACTION`: 0.5 (default - unfreeze last 50% of layers)
- `AUGMENTATION_STRENGTH`: 0.5 (default - 0.0 = none, 1.0 = max)
- `BATCH_SIZE`: 32 (default)
- `MAX_EPOCHS`: 100 (default)
- `EARLY_STOPPING_PATIENCE`: 15 (default)

**Inputs**:
- Preprocessed PNG images (512x512) from `VinDr_Optimization_Tutorial.ipynb`
- Train/val metadata CSVs

**Outputs**:
- `best_model.pt`: Trained model checkpoint
- `training_history.png`: Loss and metrics curves

**Runtime**: ~2-6 hours (depends on GPU and dataset size)

---

### 2. Simple_Inference_VinDr.ipynb

**Purpose**: Load trained model and evaluate on test set.

**Features**:
- Load trained checkpoint
- Run inference on test images
- Image-level and breast-level evaluation
- Noisy-OR aggregation for multi-view breasts
- ROC and Precision-Recall curves
- Confusion matrix and classification report
- Sample prediction visualization
- Export predictions to CSV

**Metrics Computed**:
- **Image-level**: AUROC, PR-AUC, Brier score
- **Breast-level**: AUROC, PR-AUC, Brier score (Noisy-OR aggregation)
- **Classification**: Sensitivity, Specificity, PPV, NPV

**Inputs**:
- Trained model checkpoint (`best_model.pt`)
- Test metadata CSV
- Preprocessed PNG images

**Outputs**:
- `test_predictions.csv`: Predictions for all images and breasts
- ROC and PR curve visualizations
- Classification metrics

**Runtime**: ~5-15 minutes (depends on test set size)

---

## Quick Start

### Prerequisites

1. **Preprocessed Dataset**: Run `VinDr_Optimization_Tutorial.ipynb` to convert DICOM → PNG
   - Output: `/content/drive/MyDrive/vindr-mammo/preprocessed_png_512/`
   - Includes: `train.csv`, `val.csv`, `test.csv`

2. **Google Colab** with GPU (T4, V100, or A100)

3. **Google Drive** mounted with VinDr-Mammo data

### Training Workflow

```bash
# 1. Open Simple_Training_VinDr.ipynb in Google Colab
# 2. Update paths in configuration cell
# 3. Run all cells
# 4. Wait for training to complete (~2-6 hours)
# 5. Check output: /content/drive/MyDrive/training_output/best_model.pt
```

### Inference Workflow

```bash
# 1. Open Simple_Inference_VinDr.ipynb in Google Colab
# 2. Update CHECKPOINT_PATH to point to trained model
# 3. Run all cells
# 4. Review metrics and visualizations
# 5. Check predictions: /content/drive/MyDrive/training_output/test_predictions.csv
```

---

## Model Architecture Details

### ResNet-50 Backbone

- **Pretrained**: ImageNet weights (IMAGENET1K_V1)
- **Total parameters**: ~23.5M
- **Trainable parameters**: Varies based on `UNFREEZE_FRACTION`
  - 0.0: ~2K (classifier only)
  - 0.5: ~11M (last 50% of layers + classifier)
  - 1.0: ~23.5M (all layers)

### Partial Fine-Tuning Strategy

The `UNFREEZE_FRACTION` parameter controls how many layers are fine-tuned:

- **0.0 (Feature Extraction)**: Freeze all backbone, train only classifier
  - Fastest training
  - Best when dataset is small or very different from ImageNet

- **0.5 (Partial Fine-Tuning)**: Unfreeze last 50% of layers
  - Balance between speed and performance
  - Recommended starting point

- **1.0 (Full Fine-Tuning)**: Unfreeze all layers
  - Slowest training
  - Best when dataset is large and similar to ImageNet

### Dropout Regularization

- Applied before final classification layer
- Helps prevent overfitting
- Typical range: 0.0 - 0.5

### Loss Function

- **Binary Cross-Entropy (BCE)**: Standard for binary classification
- Directly optimizes probability predictions

### Optimizer

- **AdamW**: Adam with weight decay regularization
- Decouples weight decay from gradient updates
- Better generalization than standard Adam

---

## Data Augmentation

Intensity-based augmentation for mammograms (no geometric transforms):

1. **Brightness**: Random shift ±20% × `AUGMENTATION_STRENGTH`
2. **Contrast**: Random scale ±20% × `AUGMENTATION_STRENGTH`
3. **Noise**: Gaussian noise with std 5% × `AUGMENTATION_STRENGTH`

**Why intensity-only?**
- Preserves anatomical structure
- Clinically relevant (mimics acquisition variations)
- Avoids unrealistic transformations (e.g., flipping lesions)

---

## Breast-Level Evaluation

### Why Breast-Level?

In clinical practice, diagnosis is made per breast, not per image. Each breast typically has 2-4 views (CC, MLO, etc.).

### Noisy-OR Aggregation

Combines multiple view predictions for the same breast:

```
P(breast malignant) = 1 - ∏(1 - P(view_i malignant))
```

**Intuition**: If *any* view shows cancer, the breast likely has cancer.

**Properties**:
- Monotonic: More views → higher confidence
- Asymmetric: Biased toward detecting cancer (conservative)
- Clinically motivated: Matches radiologist reasoning

---

## Hyperparameter Tuning Guide

### Learning Rate

- **Too high**: Loss oscillates, doesn't converge
- **Too low**: Slow convergence, may get stuck
- **Recommended**: 1e-4 to 1e-3 for fine-tuning, 1e-3 to 1e-2 for feature extraction

### Weight Decay

- **Too high**: Underfitting, low training accuracy
- **Too low**: Overfitting, gap between train/val metrics
- **Recommended**: 1e-5 to 1e-3

### Dropout Rate

- **Too high**: Underfitting, low capacity
- **Too low**: Overfitting on small datasets
- **Recommended**: 0.2 to 0.5 for small datasets, 0.0 to 0.3 for large datasets

### Unfreeze Fraction

- **Small dataset (<1000 images)**: 0.0 to 0.3
- **Medium dataset (1000-10000 images)**: 0.3 to 0.7
- **Large dataset (>10000 images)**: 0.7 to 1.0

### Augmentation Strength

- **Conservative (clinical)**: 0.3 to 0.5
- **Aggressive (limited data)**: 0.7 to 1.0
- **None (large dataset)**: 0.0

---

## Troubleshooting

### Training Issues

**Problem**: Validation metrics don't improve
- Check if data is properly preprocessed
- Reduce learning rate
- Increase augmentation
- Increase unfreeze fraction

**Problem**: Overfitting (train >> val)
- Increase dropout rate
- Increase weight decay
- Increase augmentation strength
- Reduce unfreeze fraction

**Problem**: Underfitting (both train and val are poor)
- Increase unfreeze fraction
- Reduce dropout rate
- Reduce weight decay
- Increase max epochs

**Problem**: Loss is NaN
- Reduce learning rate (too high)
- Check for corrupted images
- Ensure proper normalization

### Inference Issues

**Problem**: Model checkpoint not found
- Check `CHECKPOINT_PATH` is correct
- Ensure training completed successfully

**Problem**: Different results from validation
- Ensure using same preprocessing
- Check dropout is disabled (model.eval())
- Verify same aggregation method

---

## Comparison with Full Optimization

| Feature | Simple Training | Multi-Objective Optimization |
|---------|----------------|------------------------------|
| **Training time** | 2-6 hours | 5-50 days |
| **Hyperparameters** | Manual tuning | Automated search |
| **Objectives** | Single (PR-AUC) | Multiple (PR-AUC, AUROC, Brier, Robustness) |
| **Output** | Single model | Pareto front of models |
| **Complexity** | Low | High |
| **Best for** | Quick experiments, baselines | Production, research |

---

## Example Results

**Typical performance on VinDr-Mammo** (stratified subset):

### Image-Level
- AUROC: 0.75 - 0.85
- PR-AUC: 0.40 - 0.60
- Brier: 0.15 - 0.25

### Breast-Level (Noisy-OR)
- AUROC: 0.80 - 0.90
- PR-AUC: 0.50 - 0.70
- Brier: 0.10 - 0.20

*Note: Results vary based on dataset size, quality, and hyperparameters.*

---

## Citation

If you use these notebooks in your research, please cite:

```bibtex
@software{simple_mammography_training,
  title = {Simple Mammography Training with ResNet-50},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/dtobi59/mammography-multiobjective-optimization}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation in the main repository

---

## Related Files

- `VinDr_Optimization_Tutorial.ipynb`: DICOM → PNG preprocessing
- `mammogram_preprocessor.py`: Preprocessing pipeline implementation
- `models/resnet.py`: Full ResNet-50 implementation with partial fine-tuning
- `training/trainer.py`: Advanced trainer with robustness evaluation
- `optimization/nsga3_runner.py`: Multi-objective optimization
