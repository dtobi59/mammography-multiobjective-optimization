# pymoo Problem Class: Four Objectives Implementation

## Executive Summary

This document proves that the pymoo `Problem._evaluate_single()` method returns **exactly four objectives** in the correct order, and that **PR-AUC and AUROC are computed on breast-level predictions** (not image-level).

---

## 1. Objective Vector Construction

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

### Confirmed Order:
1. **Objective 1:** `-PR-AUC` (line 181) - Negated to convert maximization to minimization
2. **Objective 2:** `-AUROC` (line 182) - Negated to convert maximization to minimization
3. **Objective 3:** `Brier` (line 183) - Already a minimization metric
4. **Objective 4:** `Robustness degradation R` (line 184) - Already a minimization metric

### Context (lines 118-198):
```python
def _evaluate_single(self, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a single hyperparameter configuration.

    Args:
        x: Hyperparameter vector

    Returns:
        Objective values: [-PR-AUC, -AUROC, Brier, Robustness_degradation]
    """
    # Decode hyperparameters
    hparams = self._decode_hyperparameters(x)

    print(f"\n=== Evaluation {self.evaluation_counter + 1} ===")
    print(f"Hyperparameters: {hparams}")

    # Set random seeds for reproducibility
    set_all_seeds(config.RANDOM_SEED)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_metadata=self.train_metadata,
        val_metadata=self.val_metadata,
        image_dir=self.image_dir,
        batch_size=config.BATCH_SIZE,
        augmentation_strength=hparams["augmentation_strength"],
        num_workers=self.n_workers,
    )

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet50WithPartialFineTuning(
        unfreeze_fraction=hparams["unfreeze_fraction"],
        dropout_rate=hparams["dropout_rate"],
        pretrained=True,
    )

    # Train model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_metadata=self.val_metadata,
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        max_epochs=config.MAX_EPOCHS,
        device=device,
        checkpoint_dir=f"{self.checkpoint_dir}/eval_{self.evaluation_counter}",
    )

    best_metrics = trainer.train()

    # Evaluate robustness
    robustness_evaluator = RobustnessEvaluator(
        model=model,
        val_loader=val_loader,
        val_metadata=self.val_metadata,
        device=device,
    )
    robustness_degradation = robustness_evaluator.evaluate()

    # Compute objectives (all minimization)
    objectives = np.array([
        -best_metrics["pr_auc"],         # Maximize PR-AUC
        -best_metrics["auroc"],          # Maximize AUROC
        best_metrics["brier"],           # Minimize Brier score
        robustness_degradation,          # Minimize robustness degradation
    ])

    # Log metrics
    print(f"Objectives: PR-AUC={best_metrics['pr_auc']:.4f}, "
          f"AUROC={best_metrics['auroc']:.4f}, "
          f"Brier={best_metrics['brier']:.4f}, "
          f"Robustness={robustness_degradation:.4f}")

    self.evaluation_counter += 1

    return objectives
```

---

## 2. Objective 1: PR-AUC (Breast-Level)

### 2.1 Where best_metrics["pr_auc"] Comes From

**Source:** `best_metrics = trainer.train()` (line 168)

**File:** `training/trainer.py`
**Function:** `train()` (lines 217-272)

#### Code Flow:
```python
def train(self) -> Dict[str, float]:
    """Train model with early stopping."""
    for epoch in range(self.max_epochs):
        # Train
        train_loss = self.train_epoch()
        self.history["train_loss"].append(train_loss)

        # Validate
        val_metrics = self.validate()  # <- CALLS VALIDATE
        self.history["val_pr_auc"].append(val_metrics["pr_auc"])

        # ... early stopping logic ...

    # Return best validation metrics
    best_epoch = self.early_stopping.best_epoch
    best_metrics = {
        "pr_auc": self.history["val_pr_auc"][best_epoch],  # <- BREAST-LEVEL PR-AUC
        "auroc": self.history["val_auroc"][best_epoch],
        "brier": self.history["val_brier"][best_epoch],
        "scaled_brier": self.history["val_scaled_brier"][best_epoch],
    }

    return best_metrics
```

### 2.2 Breast-Level Aggregation in validate()

**File:** `training/trainer.py`
**Lines:** 172-201

#### Code:
```python
@torch.no_grad()
def validate(self) -> Dict[str, float]:
    """
    Validate on validation set with breast-level aggregation.

    Returns:
        Dictionary of validation metrics
    """
    self.model.eval()

    # Collect image-level predictions
    image_predictions = {}
    image_labels = {}

    for images, labels, image_ids in self.val_loader:
        images = images.to(self.device)
        predictions = self.model(images).cpu().numpy()

        for img_id, pred, label in zip(image_ids, predictions, labels.numpy()):
            image_predictions[img_id] = float(pred)
            image_labels[img_id] = int(label)

    # Aggregate to breast-level using Noisy OR
    breast_predictions, breast_labels = aggregate_to_breast_level(
        image_predictions, self.val_metadata
    )

    # Compute metrics
    metrics = compute_metrics(breast_predictions, breast_labels)

    return metrics
```

#### Key Steps:
1. **Lines 181-191:** Collect image-level predictions into dictionary
2. **Lines 193-196:** **AGGREGATE TO BREAST-LEVEL** using `aggregate_to_breast_level()`
3. **Line 199:** Compute metrics **on breast-level predictions**

### 2.3 PR-AUC Computation

**File:** `training/metrics.py`
**Lines:** 65-108

#### Code:
```python
def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted probabilities, shape (n_samples,)
        labels: Ground truth binary labels, shape (n_samples,)

    Returns:
        Dictionary of metrics:
        - pr_auc: Precision-Recall AUC
        - auroc: ROC AUC
        - brier: Brier score
    """
    # ... edge case handling ...

    metrics = {
        "pr_auc": average_precision_score(labels, predictions),  # <- PR-AUC
        "auroc": roc_auc_score(labels, predictions),
        "brier": brier,
        "brier_null": brier_null,
        "scaled_brier": scaled_brier,
    }

    return metrics
```

#### Formula:
**PR-AUC** is computed using sklearn's `average_precision_score()` (line 101)

**Input:** `breast_predictions`, `breast_labels` (from Noisy OR aggregation)

---

## 3. Objective 2: AUROC (Breast-Level)

### Source Chain:
1. `trainer.train()` → `trainer.validate()` → `aggregate_to_breast_level()` → `compute_metrics()`
2. Same flow as PR-AUC (see Section 2)

### Computation:

**File:** `training/metrics.py`
**Line:** 102

```python
"auroc": roc_auc_score(labels, predictions),
```

**Formula:** ROC AUC using sklearn's `roc_auc_score()`

**Input:** `breast_predictions`, `breast_labels` (breast-level)

**Confirmation:** ✅ AUROC is computed on **breast-level predictions**, not image-level

---

## 4. Objective 3: Brier Score (Breast-Level)

### Source Chain:
Same as PR-AUC and AUROC (see Section 2)

### Computation:

**File:** `training/metrics.py`
**Lines:** 96, 103

```python
brier = brier_score_loss(labels, predictions)

metrics = {
    # ...
    "brier": brier,
    # ...
}
```

**Formula:** Brier score using sklearn's `brier_score_loss()`

**Mathematical Definition:**
```
Brier = (1/N) * Σ(p_i - y_i)²
```

**Input:** `breast_predictions`, `breast_labels` (breast-level)

**Confirmation:** ✅ Brier score is computed on **breast-level predictions**

---

## 5. Objective 4: Robustness Degradation R (Breast-Level)

### 5.1 Where robustness_degradation Comes From

**Source:** `robustness_degradation = robustness_evaluator.evaluate()` (line 177)

**File:** `training/robustness.py`
**Lines:** 44-98

#### Code:
```python
@torch.no_grad()
def evaluate(self) -> float:
    """
    Evaluate robustness degradation with deterministic perturbations.

    Computes:
    - PR-AUC under standard inference
    - PR-AUC under perturbed inference
    - Robustness degradation R = PR-AUC_standard - PR-AUC_perturbed

    Returns:
        Robustness degradation (lower is better)
    """
    self.model.eval()

    # Collect predictions under standard and perturbed inference
    standard_predictions = {}
    perturbed_predictions = {}

    for images, labels, image_ids in self.val_loader:
        # Standard inference
        images_standard = images.to(self.device)
        preds_standard = self.model(images_standard).cpu().numpy()

        # Perturbed inference with deterministic seed per image_id
        images_perturbed = torch.stack([
            self.perturbation(img, image_id=img_id)
            for img, img_id in zip(images, image_ids)
        ]).to(self.device)
        preds_perturbed = self.model(images_perturbed).cpu().numpy()

        # Store predictions
        for img_id, pred_std, pred_pert in zip(image_ids, preds_standard, preds_perturbed):
            standard_predictions[img_id] = float(pred_std)
            perturbed_predictions[img_id] = float(pred_pert)

    # Aggregate to breast-level
    breast_preds_standard, breast_labels = aggregate_to_breast_level(
        standard_predictions, self.val_metadata
    )
    breast_preds_perturbed, _ = aggregate_to_breast_level(
        perturbed_predictions, self.val_metadata
    )

    # Compute robustness degradation
    degradation = compute_robustness_degradation(
        breast_preds_standard,
        breast_preds_perturbed,
        breast_labels,
    )

    return degradation
```

#### Key Steps:
1. **Lines 62-81:** Collect predictions under standard and perturbed inference
2. **Lines 83-89:** **AGGREGATE TO BREAST-LEVEL** for both standard and perturbed
3. **Lines 91-96:** Compute robustness degradation **on breast-level predictions**

### 5.2 Robustness Degradation Computation

**File:** `training/metrics.py`
**Lines:** 111-138

#### Code:
```python
def compute_robustness_degradation(
    predictions_standard: np.ndarray,
    predictions_perturbed: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute robustness degradation.

    Robustness degradation R = PR-AUC_standard - PR-AUC_perturbed

    Args:
        predictions_standard: Predictions under standard inference
        predictions_perturbed: Predictions under perturbed inference
        labels: Ground truth labels

    Returns:
        Robustness degradation (lower is better, can be negative)
    """
    # Handle edge cases
    if len(np.unique(labels)) < 2:
        return 0.0

    pr_auc_standard = average_precision_score(labels, predictions_standard)
    pr_auc_perturbed = average_precision_score(labels, predictions_perturbed)

    degradation = pr_auc_standard - pr_auc_perturbed

    return degradation
```

#### Formula:
```
R = PR-AUC_standard - PR-AUC_perturbed
```

**Input:**
- `breast_preds_standard` (breast-level, standard inference)
- `breast_preds_perturbed` (breast-level, perturbed inference)
- `breast_labels` (breast-level ground truth)

**Confirmation:** ✅ Robustness degradation is computed on **breast-level predictions**

---

## 6. Complete Evaluation Flow Diagram

```
Problem._evaluate_single(x)
│
├─> trainer.train()
│   │
│   └─> trainer.validate()  [called each epoch]
│       │
│       ├─> Collect image-level predictions
│       │   • image_predictions = {img_id: prob}
│       │
│       ├─> aggregate_to_breast_level(image_predictions, val_metadata)
│       │   • Returns: breast_predictions, breast_labels
│       │   • Formula: p_breast = 1 - ∏(1 - p_j)
│       │
│       └─> compute_metrics(breast_predictions, breast_labels)
│           • Returns: {"pr_auc": ..., "auroc": ..., "brier": ...}
│           • PR-AUC: average_precision_score(breast_labels, breast_predictions)
│           • AUROC: roc_auc_score(breast_labels, breast_predictions)
│           • Brier: brier_score_loss(breast_labels, breast_predictions)
│
├─> robustness_evaluator.evaluate()
│   │
│   ├─> Collect standard predictions: standard_predictions = {img_id: prob}
│   ├─> Collect perturbed predictions: perturbed_predictions = {img_id: prob}
│   │
│   ├─> aggregate_to_breast_level(standard_predictions, val_metadata)
│   │   • Returns: breast_preds_standard, breast_labels
│   │
│   ├─> aggregate_to_breast_level(perturbed_predictions, val_metadata)
│   │   • Returns: breast_preds_perturbed, _
│   │
│   └─> compute_robustness_degradation(breast_preds_standard, breast_preds_perturbed, breast_labels)
│       • R = PR-AUC_standard - PR-AUC_perturbed
│
└─> Assemble objectives
    objectives = np.array([
        -best_metrics["pr_auc"],      # Obj 1: -PR-AUC (breast-level)
        -best_metrics["auroc"],       # Obj 2: -AUROC (breast-level)
        best_metrics["brier"],        # Obj 3: Brier (breast-level)
        robustness_degradation,       # Obj 4: R (breast-level)
    ])
```

---

## 7. Summary Table: Four Objectives

| Index | Objective | Sign | Optimization | Breast-Level? | File | Line | Function |
|-------|-----------|------|--------------|---------------|------|------|----------|
| 0 | PR-AUC | Negated (-) | Maximize | ✅ Yes | `training/metrics.py` | 101 | `average_precision_score()` |
| 1 | AUROC | Negated (-) | Maximize | ✅ Yes | `training/metrics.py` | 102 | `roc_auc_score()` |
| 2 | Brier | Positive (+) | Minimize | ✅ Yes | `training/metrics.py` | 96 | `brier_score_loss()` |
| 3 | Robustness R | Positive (+) | Minimize | ✅ Yes | `training/metrics.py` | 136 | `pr_auc_std - pr_auc_pert` |

---

## 8. Proof of Breast-Level Computation

### Evidence 1: Trainer Validation

**File:** `training/trainer.py`
**Lines:** 193-199

```python
# Aggregate to breast-level using Noisy OR
breast_predictions, breast_labels = aggregate_to_breast_level(
    image_predictions, self.val_metadata
)

# Compute metrics
metrics = compute_metrics(breast_predictions, breast_labels)
```

**Proof:**
- Input to `compute_metrics()` is `breast_predictions` (not `image_predictions`)
- Therefore, PR-AUC, AUROC, and Brier are computed on breast-level data

### Evidence 2: Robustness Evaluation

**File:** `training/robustness.py`
**Lines:** 83-96

```python
# Aggregate to breast-level
breast_preds_standard, breast_labels = aggregate_to_breast_level(
    standard_predictions, self.val_metadata
)
breast_preds_perturbed, _ = aggregate_to_breast_level(
    perturbed_predictions, self.val_metadata
)

# Compute robustness degradation
degradation = compute_robustness_degradation(
    breast_preds_standard,
    breast_preds_perturbed,
    breast_labels,
)
```

**Proof:**
- Inputs to `compute_robustness_degradation()` are `breast_preds_*` (not `image_preds_*`)
- Therefore, robustness degradation is computed on breast-level data

---

## 9. Early Stopping Criterion

**File:** `training/trainer.py`
**Lines:** 124-128

```python
# Early stopping
self.early_stopping = EarlyStopping(
    patience=config.EARLY_STOPPING_PATIENCE,
    mode="max",  # Maximize PR-AUC
)
```

**Confirmation:** Early stopping uses **breast-level PR-AUC** as the criterion (from `trainer.validate()`)

---

## 10. Code Location Reference

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **Objective construction** | `optimization/problem.py` | 180-185 | Returns 4-element array |
| **Trainer validation** | `training/trainer.py` | 172-201 | Aggregates to breast-level |
| **Metric computation** | `training/metrics.py` | 65-108 | Computes PR-AUC, AUROC, Brier |
| **Robustness evaluation** | `training/robustness.py` | 44-98 | Computes R on breast-level |
| **Robustness degradation** | `training/metrics.py` | 111-138 | R = PR-AUC_std - PR-AUC_pert |
| **Noisy OR aggregation** | `utils/noisy_or.py` | 26-83 | Image-level → breast-level |

---

## Conclusion

**CONFIRMED:**

1. ✅ The pymoo `Problem._evaluate_single()` method returns **exactly four objectives**
2. ✅ The objectives are in the **exact order**:
   - Index 0: `-PR-AUC`
   - Index 1: `-AUROC`
   - Index 2: `Brier`
   - Index 3: `Robustness degradation R`
3. ✅ **PR-AUC and AUROC are computed on breast-level predictions**, not image-level
4. ✅ **Brier score is also computed on breast-level predictions**
5. ✅ **Robustness degradation R is computed on breast-level predictions**

**All four objectives are evaluated at the breast level after Noisy OR aggregation.**
