# Implementation Changes for Research-Grade Pipeline

## A) Code Changes Summary

### 1. **data/dataset.py** - Stratified Subsampling with Manifest
**Function:** `create_stratified_subsample()`
- **Input:** Full VinDr metadata
- **Output:** (train_metadata, val_metadata, manifest_path)
- **Logic:**
  1. Exclude BI-RADS 3
  2. Patient-wise split (80/20) using existing `create_train_val_split()`
  3. Within each partition, apply stratified sampling:
     - Train: ~200 malignant, ~600 benign (80% of 250/750)
     - Val: ~50 malignant, ~150 benign (20% of 250/750)
  4. Save manifest CSV with columns:
     - image_id, patient_id, breast_id, laterality, view, birads_original, label, split, image_path
  5. Log final counts (images and breasts)

### 2. **utils/noisy_or.py** - Generalized Noisy OR
**Function:** `aggregate_to_breast_level()`
- **Change:** Remove assumption of exactly 2 views (CC + MLO)
- **New logic:**
  ```python
  # For each breast_id, collect all available image predictions
  image_probs = [image_predictions[img_id] for img_id in group["image_id"]]

  # Generalized Noisy OR: p_breast = 1 - ∏(1 - p_j)
  p_breast = 1.0 - np.prod([1.0 - p for p in image_probs])

  # Special case: if only 1 image, p_breast = p_1
  ```
- **Handles:** 1 view, 2 views, or more per breast

### 3. **training/metrics.py** - Scaled Brier Score
**New functions:**
```python
def compute_brier_null(labels: np.ndarray) -> float:
    """
    Compute null Brier score (predicting prevalence for all samples).

    Brier_null = mean((prevalence - y_i)^2)
    """
    prevalence = labels.mean()
    return np.mean((prevalence - labels) ** 2)

def compute_scaled_brier(brier: float, labels: np.ndarray) -> float:
    """
    Compute Scaled Brier score.

    Scaled Brier = 1 - (Brier / Brier_null)
    Range: (-∞, 1], where 1 is perfect, 0 is no better than prevalence
    """
    brier_null = compute_brier_null(labels)
    if brier_null == 0:
        return 0.0
    return 1.0 - (brier / brier_null)
```

**Update:** `compute_metrics()` to return Scaled Brier

### 4. **training/robustness.py** - Deterministic Perturbations
**Change:** In `RobustnessEvaluator.evaluate()`
- **Old:** `seed=config.RANDOM_SEED + i` (batch index)
- **New:** `seed=hash(image_id) % (2**31)` (derived from image_id)
- **Result:** Same image always gets same perturbation, reproducible across runs

### 5. **data/augmentation.py** - Fix Perturbation Seed
**Update:** `RobustnessPerturbation.__call__()`
- **Add parameter:** `image_id: str` instead of generic `seed`
- **Compute seed:** `seed = hash(image_id) % (2**31 - 1)` inside function
- **Ensure:** Deterministic per image_id

### 6. **optimization/nsga3_runner.py** - Use New Subsampling
**Changes:**
```python
# Replace this section in __main__ or wherever data loading happens:
from data.dataset import create_stratified_subsample

# Load and subsample
train_metadata, val_metadata, manifest_path = create_stratified_subsample(
    vindr_metadata,
    target_total=1000,
    target_malignant=250,
    target_benign=750,
    train_ratio=0.8,
    random_seed=config.RANDOM_SEED,
    output_dir="./manifests"
)

# Log counts
print("=" * 80)
print("DATASET SUBSAMPLING SUMMARY")
print("=" * 80)
print(f"Train: {len(train_metadata)} images, {train_metadata['breast_id'].nunique()} breasts")
print(f"  Malignant: {(train_metadata['label'] == 1).sum()}")
print(f"  Benign: {(train_metadata['label'] == 0).sum()}")
print(f"Val: {len(val_metadata)} images, {val_metadata['breast_id'].nunique()} breasts")
print(f"  Malignant: {(val_metadata['label'] == 1).sum()}")
print(f"  Benign: {(val_metadata['label'] == 0).sum()}")
print(f"Manifest saved to: {manifest_path}")
print("=" * 80)
```

### 7. **optimization/problem.py** - Log Scaled Brier
**Changes:**
```python
# In _evaluate_single(), after computing metrics:
best_metrics = trainer.train()

# Add Scaled Brier to logging (NOT to objectives)
print(f"Objectives: PR-AUC={best_metrics['pr_auc']:.4f}, "
      f"AUROC={best_metrics['auroc']:.4f}, "
      f"Brier={best_metrics['brier']:.4f}, "
      f"Scaled Brier={best_metrics.get('scaled_brier', 0.0):.4f}, "
      f"Robustness={robustness_degradation:.4f}")

# Objectives remain unchanged (4 objectives only):
objectives = np.array([
    -best_metrics["pr_auc"],
    -best_metrics["auroc"],
    best_metrics["brier"],
    robustness_degradation,
])
```

### 8. **training/trainer.py** - Return Scaled Brier
**Change:** In `validate()` method, add Scaled Brier to returned metrics:
```python
# Compute metrics
metrics = compute_metrics(breast_predictions, breast_labels)

# Add Scaled Brier
from training.metrics import compute_scaled_brier
metrics['scaled_brier'] = compute_scaled_brier(metrics['brier'], breast_labels)

return metrics
```

---

## B) Files to Modify

| File | Changes | Reason |
|------|---------|--------|
| `data/dataset.py` | Add `create_stratified_subsample()` | Implement stratified sampling with manifest |
| `utils/noisy_or.py` | Update `aggregate_to_breast_level()` | Handle variable number of views |
| `training/metrics.py` | Add `compute_brier_null()`, `compute_scaled_brier()`, update `compute_metrics()` | Add Scaled Brier metric |
| `training/robustness.py` | Change seed derivation from batch index to image_id | Make perturbations deterministic |
| `data/augmentation.py` | Update `RobustnessPerturbation.__call__()` | Accept image_id for seed derivation |
| `training/trainer.py` | Add Scaled Brier to returned metrics | Include in validation output |
| `optimization/problem.py` | Add Scaled Brier logging | Report (not optimize) Scaled Brier |
| `optimization/nsga3_runner.py` | Use new subsampling, add logging | Apply changes to main workflow |

---

## C) Testing Checklist

- [ ] Stratified sampling produces ~250/750 split
- [ ] Patient-wise splitting has NO leakage (no patient in both train and val)
- [ ] Manifest saved correctly with all required columns
- [ ] Noisy OR works with 1 view per breast
- [ ] Noisy OR works with 2 views per breast
- [ ] Scaled Brier computed correctly
- [ ] Robustness perturbations are deterministic (same image_id → same perturbation)
- [ ] 4 objectives still optimized (Scaled Brier is only logged)
- [ ] Train/val image and breast counts logged correctly
- [ ] Full optimization pipeline runs end-to-end

---

## D) Key Design Decisions

1. **Patient-wise split FIRST:** Prevents data leakage. Stratification happens within each partition.

2. **Generalized Noisy OR:** Handles subsampling where some breasts have only 1 view.

3. **Deterministic perturbations:** Uses `hash(image_id)` to ensure same perturbation for same image across evaluations.

4. **Scaled Brier is logged, not optimized:** Keeps 4 objectives, adds interpretability.

5. **Manifest saved to disk:** Reproducibility and transparency for research.

---

## E) Expected Output

After implementation, when running optimization:

```
================================================================================
DATASET SUBSAMPLING SUMMARY
================================================================================
Excluding BI-RADS 3: 18,234 images remaining
Train: 800 images, 423 breasts
  Malignant: 200
  Benign: 600
Val: 200 images, 98 breasts
  Malignant: 50
  Benign: 150
Manifest saved to: ./manifests/vindr_subsample_seed42_20260104.csv
================================================================================

=== Evaluation 1 ===
Hyperparameters: {'learning_rate': 0.001, 'weight_decay': 1e-05, ...}
Epoch 1/100 - Loss: 0.4523, Val PR-AUC: 0.7234, Val AUROC: 0.8123, Val Brier: 0.1456
...
Objectives: PR-AUC=0.8234, AUROC=0.9012, Brier=0.1234, Scaled Brier=0.4567, Robustness=0.0234
```

---

## F) Backward Compatibility

- Existing evaluation scripts (`evaluate_source.py`, `evaluate_target.py`) will work with updated Noisy OR
- INbreast dataset can still be evaluated zero-shot
- Checkpoint format unchanged
- NSGA-III optimization loop unchanged
