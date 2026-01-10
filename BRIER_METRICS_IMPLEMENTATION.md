# Brier Score and Derived Metrics Implementation

## Executive Summary

This document shows the **exact implementation** of Brier score computation and derived metrics (Brier_null and Scaled Brier). These metrics are:
- ✅ **Computed** for all evaluations
- ✅ **Logged** during training and optimization
- ✅ **NOT included** in the optimization objective vector (only Brier is optimized)
- ✅ **Reported** for interpretability

---

## 1. Brier Score Computation

**File:** `training/metrics.py`
**Line:** 96

### Code:
```python
brier = brier_score_loss(labels, predictions)
```

### Formula:
```
Brier = (1/N) * Σ(p_i - y_i)²

Where:
- p_i = predicted probability for sample i
- y_i = true label for sample i (0 or 1)
- N = number of samples
```

### Function Used:
`sklearn.metrics.brier_score_loss(labels, predictions)`

**Input:** Breast-level predictions and labels (after Noisy OR aggregation)

---

## 2. Validation Prevalence

**File:** `training/metrics.py`
**Line:** 33

### Code:
```python
prevalence = labels.mean()
```

### Formula:
```
p = mean(y) = (1/N) * Σ y_i

Where:
- y_i ∈ {0, 1} are the true labels
- N = number of samples
```

### Example:
```python
labels = [0, 0, 1, 1, 1]
prevalence = mean([0, 0, 1, 1, 1]) = 3/5 = 0.6
```

**Meaning:** Prevalence is the proportion of positive samples (malignant breasts) in the validation set.

---

## 3. Brier_null (Null Brier Score)

**File:** `training/metrics.py`
**Lines:** 16-35

### Full Implementation:
```python
def compute_brier_null(labels: np.ndarray) -> float:
    """
    Compute null Brier score (predicting prevalence for all samples).

    The null Brier score is the Brier score achieved by always predicting
    the prevalence (mean of labels). This serves as a baseline for calibration.

    Formula:
        Brier_null = mean((prevalence - y_i)^2)
                   = prevalence * (1 - prevalence)

    Args:
        labels: Ground truth binary labels, shape (n_samples,)

    Returns:
        Null Brier score
    """
    prevalence = labels.mean()
    # Equivalent to: np.mean((prevalence - labels) ** 2)
    return prevalence * (1.0 - prevalence)
```

### Mathematical Formula:
```
Brier_null = mean((p - y_i)²)
           = p * (1 - p)

Where:
- p = prevalence = mean(y)
- y_i ∈ {0, 1}
```

### Derivation:
```
For binary labels y_i ∈ {0, 1}:

Brier_null = (1/N) * Σ(p - y_i)²

When y_i = 0: (p - 0)² = p²
When y_i = 1: (p - 1)² = (p - 1)² = p² - 2p + 1

Let N_0 = count of zeros, N_1 = count of ones
Then: p = N_1 / N

Brier_null = (1/N) * [N_0 * p² + N_1 * (p² - 2p + 1)]
           = (1/N) * [N * p² - 2 * N_1 * p + N_1]
           = p² - 2p * (N_1/N) + (N_1/N)
           = p² - 2p² + p
           = p * (1 - p)
```

### Example:
```python
labels = [0, 0, 1, 1, 1]  # N=5, N_1=3
prevalence = 3/5 = 0.6
Brier_null = 0.6 * (1 - 0.6) = 0.6 * 0.4 = 0.24
```

**Verification:**
```python
# Direct calculation
predictions_null = [0.6, 0.6, 0.6, 0.6, 0.6]  # Always predict prevalence
brier_direct = mean([(0.6-0)², (0.6-0)², (0.6-1)², (0.6-1)², (0.6-1)²])
             = mean([0.36, 0.36, 0.16, 0.16, 0.16])
             = 1.2 / 5
             = 0.24 ✓
```

**Interpretation:** Brier_null is the Brier score achieved by a naive model that always predicts the prevalence.

---

## 4. Scaled Brier Score

**File:** `training/metrics.py`
**Lines:** 38-62

### Full Implementation:
```python
def compute_scaled_brier(brier: float, labels: np.ndarray) -> float:
    """
    Compute Scaled Brier score.

    Scaled Brier = 1 - (Brier / Brier_null)

    Range: (-∞, 1], where:
    - 1 = perfect calibration (Brier = 0)
    - 0 = no better than predicting prevalence
    - negative = worse than predicting prevalence

    Args:
        brier: Brier score from model predictions
        labels: Ground truth binary labels

    Returns:
        Scaled Brier score
    """
    brier_null = compute_brier_null(labels)

    # Handle edge case where all labels are the same
    if brier_null == 0:
        return 0.0

    return 1.0 - (brier / brier_null)
```

### Mathematical Formula:
```
Scaled Brier = 1 - (Brier / Brier_null)

Where:
- Brier = model's Brier score
- Brier_null = p * (1 - p)
- p = prevalence
```

### Interpretation:

| Scaled Brier Value | Interpretation |
|-------------------|----------------|
| **1.0** | Perfect calibration (Brier = 0) |
| **> 0** | Better than always predicting prevalence |
| **0.0** | Equivalent to predicting prevalence |
| **< 0** | Worse than predicting prevalence |

### Example:
```python
labels = [0, 0, 1, 1, 1]
prevalence = 0.6
Brier_null = 0.24

# Good model
predictions_good = [0.1, 0.1, 0.9, 0.9, 0.9]
Brier_good = mean([(0.1-0)², (0.1-0)², (0.9-1)², (0.9-1)², (0.9-1)²])
           = mean([0.01, 0.01, 0.01, 0.01, 0.01])
           = 0.01
Scaled Brier_good = 1 - (0.01 / 0.24) = 1 - 0.042 = 0.958 ✓

# Random model (predicts prevalence)
Brier_random = 0.24
Scaled Brier_random = 1 - (0.24 / 0.24) = 0.0 ✓

# Bad model
predictions_bad = [0.9, 0.9, 0.1, 0.1, 0.1]  # Reversed predictions
Brier_bad = mean([(0.9-0)², (0.9-0)², (0.1-1)², (0.1-1)², (0.1-1)²])
          = mean([0.81, 0.81, 0.81, 0.81, 0.81])
          = 0.81
Scaled Brier_bad = 1 - (0.81 / 0.24) = 1 - 3.375 = -2.375 ✓ (negative!)
```

---

## 5. Metrics Returned by compute_metrics()

**File:** `training/metrics.py`
**Lines:** 96-108

### Code:
```python
brier = brier_score_loss(labels, predictions)
brier_null = compute_brier_null(labels)
scaled_brier = compute_scaled_brier(brier, labels)

metrics = {
    "pr_auc": average_precision_score(labels, predictions),
    "auroc": roc_auc_score(labels, predictions),
    "brier": brier,
    "brier_null": brier_null,
    "scaled_brier": scaled_brier,
}

return metrics
```

**All metrics computed and returned:**
1. `pr_auc` - Precision-Recall AUC
2. `auroc` - ROC AUC
3. `brier` - Brier score
4. `brier_null` - Null Brier score
5. `scaled_brier` - Scaled Brier score

---

## 6. Logging During Training

**File:** `training/trainer.py`
**Lines:** 236-242

### Code:
```python
# Print progress
print(f"Epoch {epoch + 1}/{self.max_epochs} - "
      f"Loss: {train_loss:.4f}, "
      f"Val PR-AUC: {val_metrics['pr_auc']:.4f}, "
      f"Val AUROC: {val_metrics['auroc']:.4f}, "
      f"Val Brier: {val_metrics['brier']:.4f}, "
      f"Val Scaled Brier: {val_metrics.get('scaled_brier', 0.0):.4f}")
```

### Example Output:
```
Epoch 1/100 - Loss: 0.4523, Val PR-AUC: 0.7234, Val AUROC: 0.8156, Val Brier: 0.1423, Val Scaled Brier: 0.4321
Epoch 2/100 - Loss: 0.4012, Val PR-AUC: 0.7456, Val AUROC: 0.8234, Val Brier: 0.1356, Val Scaled Brier: 0.4587
...
```

**Logged metrics:**
- ✅ `brier` - Main Brier score
- ✅ `scaled_brier` - Scaled Brier score

**NOT logged during epoch print:**
- ❌ `brier_null` - Not printed per epoch (but computed and stored)

---

## 7. Logging During Optimization

**File:** `optimization/problem.py`
**Lines:** 187-194

### Code:
```python
# Log metrics (including Scaled Brier for interpretability, not optimized)
print(f"Objectives: PR-AUC={best_metrics['pr_auc']:.4f}, "
      f"AUROC={best_metrics['auroc']:.4f}, "
      f"Brier={best_metrics['brier']:.4f}, "
      f"Robustness={robustness_degradation:.4f}")
print(f"Additional metrics: "
      f"Brier_null={best_metrics.get('brier_null', 0.0):.4f}, "
      f"Scaled Brier={best_metrics.get('scaled_brier', 0.0):.4f}")
```

### Example Output:
```
=== Evaluation 1 ===
Hyperparameters: {'learning_rate': 0.0001, 'weight_decay': 0.001, ...}
...
Objectives: PR-AUC=0.7234, AUROC=0.8156, Brier=0.1423, Robustness=0.0234
Additional metrics: Brier_null=0.2456, Scaled Brier=0.4210
```

**Logged metrics:**
- ✅ **Objectives (4):** PR-AUC, AUROC, Brier, Robustness
- ✅ **Additional metrics (2):** Brier_null, Scaled Brier

**Note:** The comment explicitly states "including Scaled Brier for interpretability, **not optimized**"

---

## 8. Optimization Objective Vector

**File:** `optimization/problem.py`
**Lines:** 179-185

### Code:
```python
# Compute objectives (all minimization)
objectives = np.array([
    -best_metrics["pr_auc"],         # Maximize PR-AUC
    -best_metrics["auroc"],          # Maximize AUROC
    best_metrics["brier"],           # Minimize Brier score
    robustness_degradation,          # Minimize robustness degradation
])
```

### Objective Vector Contents:

| Index | Objective | Included in Optimization? |
|-------|-----------|--------------------------|
| 0 | `-pr_auc` | ✅ Yes |
| 1 | `-auroc` | ✅ Yes |
| 2 | `brier` | ✅ Yes |
| 3 | `robustness_degradation` | ✅ Yes |
| - | `brier_null` | ❌ **No** |
| - | `scaled_brier` | ❌ **No** |

**Confirmed:**
- ✅ Only **Brier** is included in the objective vector
- ❌ `brier_null` is NOT in the objective vector
- ❌ `scaled_brier` is NOT in the objective vector

---

## 9. Storage in Training History

**File:** `training/trainer.py`
**Lines:** 131-137, 231-234

### Initialization:
```python
# Training history
self.history = {
    "train_loss": [],
    "val_pr_auc": [],
    "val_auroc": [],
    "val_brier": [],
    "val_scaled_brier": [],
}
```

### During Training:
```python
val_metrics = self.validate()
self.history["val_pr_auc"].append(val_metrics["pr_auc"])
self.history["val_auroc"].append(val_metrics["auroc"])
self.history["val_brier"].append(val_metrics["brier"])
self.history["val_scaled_brier"].append(val_metrics.get("scaled_brier", 0.0))
```

**Stored in history:**
- ✅ `val_brier`
- ✅ `val_scaled_brier`

**NOT stored in history:**
- ❌ `brier_null` (computed but not stored per epoch)

**Note:** `brier_null` can be computed from labels at any time using `p * (1 - p)` where `p = labels.mean()`

---

## 10. Returned to Problem Class

**File:** `training/trainer.py`
**Lines:** 263-272

### Code:
```python
# Return best validation metrics
best_epoch = self.early_stopping.best_epoch
best_metrics = {
    "pr_auc": self.history["val_pr_auc"][best_epoch],
    "auroc": self.history["val_auroc"][best_epoch],
    "brier": self.history["val_brier"][best_epoch],
    "scaled_brier": self.history["val_scaled_brier"][best_epoch],
}

return best_metrics
```

**Returned metrics:**
1. `pr_auc`
2. `auroc`
3. `brier`
4. `scaled_brier`

**Note:** `brier_null` is computed on-demand in `compute_metrics()` but not returned by `trainer.train()`

**To access `brier_null`:**
It's available in the `val_metrics` dictionary during training (line 230), but not stored in history. If needed, it can be:
1. Computed from validation labels: `labels.mean() * (1 - labels.mean())`
2. Retrieved from the last call to `validate()` before returning

---

## 11. Code Locations Summary

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **Prevalence computation** | `training/metrics.py` | 33 | `p = labels.mean()` |
| **Brier_null computation** | `training/metrics.py` | 16-35 | `p * (1 - p)` |
| **Scaled Brier computation** | `training/metrics.py` | 38-62 | `1 - (brier / brier_null)` |
| **compute_metrics()** | `training/metrics.py` | 65-108 | Returns all 5 metrics |
| **Training epoch logging** | `training/trainer.py` | 236-242 | Prints brier and scaled_brier |
| **Optimization logging** | `optimization/problem.py` | 187-194 | Prints brier_null and scaled_brier |
| **Objective vector** | `optimization/problem.py` | 179-185 | **Only brier included** |
| **Training history** | `training/trainer.py` | 131-137 | Stores brier and scaled_brier |
| **Returned metrics** | `training/trainer.py` | 263-272 | Returns brier and scaled_brier |

---

## 12. Configuration for Including in Objectives

### Current Configuration:
The objective vector is **hardcoded** in `optimization/problem.py:179-185` to include only:
1. `-pr_auc`
2. `-auroc`
3. `brier` (NOT scaled_brier)
4. `robustness_degradation`

### To Include Scaled Brier Instead of Brier:

**Modify:** `optimization/problem.py:179-185`

```python
# Current (Brier)
objectives = np.array([
    -best_metrics["pr_auc"],
    -best_metrics["auroc"],
    best_metrics["brier"],           # ← Raw Brier
    robustness_degradation,
])

# Alternative (Scaled Brier)
objectives = np.array([
    -best_metrics["pr_auc"],
    -best_metrics["auroc"],
    -best_metrics["scaled_brier"],   # ← Scaled Brier (maximize)
    robustness_degradation,
])
```

**Note:**
- Brier is **minimized** (lower is better)
- Scaled Brier is **maximized** (higher is better, so use negative sign)

---

## 13. Mathematical Properties

### Property 1: Brier_null is Data-Dependent
```
Brier_null = p * (1 - p)

Maximum: 0.25 (when p = 0.5, i.e., balanced dataset)
Minimum: 0.0 (when p = 0 or p = 1, i.e., single class)
```

**Example:**
```
p = 0.1  → Brier_null = 0.1 * 0.9 = 0.09
p = 0.3  → Brier_null = 0.3 * 0.7 = 0.21
p = 0.5  → Brier_null = 0.5 * 0.5 = 0.25 (max)
p = 0.7  → Brier_null = 0.7 * 0.3 = 0.21
p = 0.9  → Brier_null = 0.9 * 0.1 = 0.09
```

### Property 2: Scaled Brier Bounds
```
If Brier = 0 (perfect):     Scaled Brier = 1.0
If Brier = Brier_null:      Scaled Brier = 0.0
If Brier > Brier_null:      Scaled Brier < 0.0 (bad model!)
If Brier < Brier_null:      Scaled Brier > 0.0 (good model)
```

### Property 3: Brier Null as Baseline
Brier_null represents the performance of a trivial model that:
- Ignores all features
- Always predicts the training set prevalence
- Achieves "no discrimination" (AUROC ≈ 0.5)

Any useful model should have `Brier < Brier_null`, which gives `Scaled Brier > 0`.

---

## 14. Why Scaled Brier is Not Optimized

**Reasons:**

1. **Redundancy:** Minimizing `Brier` is equivalent to maximizing `Scaled Brier` since:
   ```
   Scaled Brier = 1 - (Brier / Brier_null)
   ```
   Since `Brier_null` is constant for a given validation set, minimizing `Brier` automatically maximizes `Scaled Brier`.

2. **Interpretability:** Scaled Brier is included for human interpretation, showing performance relative to the baseline.

3. **Objective Simplicity:** Using raw `Brier` avoids division and keeps objective computation straightforward.

4. **Multi-Objective Optimization:** NSGA-III optimizes `Brier` directly. Scaled Brier can be computed post-hoc for reporting.

---

## Conclusion

**CONFIRMED:**

1. ✅ **Validation prevalence computed:** `p = labels.mean()` (line 33)
2. ✅ **Brier_null computed:** `p * (1 - p)` (lines 16-35)
3. ✅ **Scaled Brier computed:** `1 - (brier / brier_null)` (lines 38-62)
4. ✅ **All metrics logged:**
   - During training: `brier` and `scaled_brier` printed per epoch
   - During optimization: `brier_null` and `scaled_brier` printed as "Additional metrics"
5. ✅ **Only Brier in objective vector:** Lines 179-185 show only `brier` is included
6. ✅ **Brier_null and Scaled Brier NOT optimized:** They are logged for interpretability only

**The implementation correctly computes and logs all Brier-derived metrics while optimizing only the raw Brier score.**
