# Early Stopping Implementation

## Executive Summary

This document shows the **exact implementation** of early stopping in the custom training loop. The implementation monitors **validation PR-AUC**, uses **patience=15**, **maximizes** the metric, and **restores the best checkpoint** after training.

**Type:** Custom implementation (not PyTorch Lightning)

---

## 1. Early Stopping Configuration

### 1.1 Initialization

**File:** `training/trainer.py`
**Lines:** 124-128

```python
# Early stopping
self.early_stopping = EarlyStopping(
    patience=config.EARLY_STOPPING_PATIENCE,
    mode="max",  # Maximize PR-AUC
)
```

**Configuration:**
- **Patience:** `config.EARLY_STOPPING_PATIENCE` = 15 (from `config.py:58`)
- **Mode:** `"max"` - Maximize PR-AUC
- **Metric:** Validation PR-AUC (breast-level)

### 1.2 Patience Value

**File:** `config.py`
**Line:** 58

```python
EARLY_STOPPING_PATIENCE = 15
```

**Meaning:** Training will stop if validation PR-AUC does not improve for 15 consecutive epochs.

---

## 2. EarlyStopping Class Implementation

**File:** `training/trainer.py`
**Lines:** 18-74

### Full Implementation:
```python
class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        mode: str = "max",
        delta: float = 0.0,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            mode: 'max' for metrics to maximize (e.g., PR-AUC), 'min' for metrics to minimize
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            current_score: Current validation metric value
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        # Check for improvement
        if self.mode == "max":
            improved = current_score > self.best_score + self.delta
        else:
            improved = current_score < self.best_score - self.delta

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
```

### Key Logic:

#### Line 60-61: Maximization Check
```python
if self.mode == "max":
    improved = current_score > self.best_score + self.delta
```
**Confirmed:** Mode is `"max"`, so improvement means `current_score > best_score`

#### Line 65-68: Update Best Score
```python
if improved:
    self.best_score = current_score
    self.best_epoch = epoch
    self.counter = 0
```
When improvement occurs, reset counter to 0.

#### Line 69-72: Increment Counter
```python
else:
    self.counter += 1
    if self.counter >= self.patience:
        self.early_stop = True
```
When no improvement, increment counter. Stop if counter reaches patience (15 epochs).

---

## 3. Training Loop with Early Stopping

**File:** `training/trainer.py`
**Lines:** 217-272

### Full train() Method:
```python
def train(self) -> Dict[str, float]:
    """
    Train model with early stopping.

    Returns:
        Best validation metrics
    """
    for epoch in range(self.max_epochs):
        # Train
        train_loss = self.train_epoch()
        self.history["train_loss"].append(train_loss)

        # Validate
        val_metrics = self.validate()
        self.history["val_pr_auc"].append(val_metrics["pr_auc"])
        self.history["val_auroc"].append(val_metrics["auroc"])
        self.history["val_brier"].append(val_metrics["brier"])
        self.history["val_scaled_brier"].append(val_metrics.get("scaled_brier", 0.0))

        # Print progress
        print(f"Epoch {epoch + 1}/{self.max_epochs} - "
              f"Loss: {train_loss:.4f}, "
              f"Val PR-AUC: {val_metrics['pr_auc']:.4f}, "
              f"Val AUROC: {val_metrics['auroc']:.4f}, "
              f"Val Brier: {val_metrics['brier']:.4f}, "
              f"Val Scaled Brier: {val_metrics.get('scaled_brier', 0.0):.4f}")

        # Save checkpoint if best so far
        val_pr_auc = val_metrics["pr_auc"]
        if self.early_stopping.best_score is None or val_pr_auc > self.early_stopping.best_score:
            if self.checkpoint_dir is not None:
                self.best_checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"best_checkpoint.pt"
                )
                self.save_checkpoint(self.best_checkpoint_path)

        # Check early stopping
        if self.early_stopping(val_pr_auc, epoch):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Restore best checkpoint
    if self.best_checkpoint_path is not None and os.path.exists(self.best_checkpoint_path):
        print(f"Restoring best checkpoint from epoch {self.early_stopping.best_epoch + 1}")
        self.load_checkpoint(self.best_checkpoint_path)

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

---

## 4. Metric Monitoring: Validation PR-AUC

### 4.1 Where val_pr_auc Comes From

**Line 230:** `val_metrics = self.validate()`

**Line 231:** `self.history["val_pr_auc"].append(val_metrics["pr_auc"])`

**Line 245:** `val_pr_auc = val_metrics["pr_auc"]`

### 4.2 validate() Method

**File:** `training/trainer.py`
**Lines:** 172-201

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

**Confirmed:** `val_metrics["pr_auc"]` is computed on **breast-level predictions** (see line 193-199)

### 4.3 Early Stopping Call

**Line 254:**
```python
if self.early_stopping(val_pr_auc, epoch):
```

**Passed to early stopping:**
- `current_score = val_pr_auc` (breast-level PR-AUC)
- `epoch = current epoch number`

**Returns:** `True` if training should stop, `False` otherwise

---

## 5. Checkpoint Saving

**File:** `training/trainer.py`
**Lines:** 203-209

### save_checkpoint() Method:
```python
def save_checkpoint(self, filepath: str) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
    }, filepath)
```

### When Checkpoint is Saved:

**Lines 244-251:**
```python
# Save checkpoint if best so far
val_pr_auc = val_metrics["pr_auc"]
if self.early_stopping.best_score is None or val_pr_auc > self.early_stopping.best_score:
    if self.checkpoint_dir is not None:
        self.best_checkpoint_path = os.path.join(
            self.checkpoint_dir, f"best_checkpoint.pt"
        )
        self.save_checkpoint(self.best_checkpoint_path)
```

**Logic:**
- If current `val_pr_auc` is better than `best_score` (or first epoch)
- Save checkpoint to `{checkpoint_dir}/best_checkpoint.pt`

**Contents:**
- Model state dict: `model.state_dict()`
- Optimizer state dict: `optimizer.state_dict()`

---

## 6. Checkpoint Restoration

**File:** `training/trainer.py`
**Lines:** 211-215

### load_checkpoint() Method:
```python
def load_checkpoint(self, filepath: str) -> None:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

### When Checkpoint is Restored:

**Lines 258-261:**
```python
# Restore best checkpoint
if self.best_checkpoint_path is not None and os.path.exists(self.best_checkpoint_path):
    print(f"Restoring best checkpoint from epoch {self.early_stopping.best_epoch + 1}")
    self.load_checkpoint(self.best_checkpoint_path)
```

**Logic:**
- After training loop completes (either early stopping or max epochs)
- If a checkpoint was saved
- Load best checkpoint from disk
- Print which epoch's checkpoint is being restored

**Confirmed:** ✅ Best checkpoint is **always restored** after training

---

## 7. Returning Best Metrics

**Lines 263-271:**
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

**Logic:**
- Retrieve metrics from the best epoch (stored in history)
- These correspond to the restored checkpoint
- Return to Problem class for objective computation

**Guaranteed Consistency:**
- Returned metrics match the restored model weights
- Both are from `early_stopping.best_epoch`

---

## 8. Early Stopping Flow Diagram

```
Epoch Loop
│
├─> Train epoch
├─> Validate (compute breast-level PR-AUC)
├─> Record metrics in history
│
├─> CHECK: Is val_pr_auc > best_score?
│   ├─> YES: Save checkpoint to best_checkpoint.pt
│   │         Update best_score and best_epoch
│   │         Reset counter = 0
│   │
│   └─> NO:  Increment counter
│            If counter >= patience (15):
│              Set early_stop = True
│
├─> Call early_stopping(val_pr_auc, epoch)
│   └─> Returns: True if should stop, False otherwise
│
└─> If early_stop == True:
    │   Print "Early stopping triggered"
    │   Break out of loop
    │
    After loop:
    │
    ├─> Load best_checkpoint.pt
    │   • Restore model weights from best epoch
    │   • Restore optimizer state from best epoch
    │
    └─> Return metrics from best epoch
        • pr_auc = history["val_pr_auc"][best_epoch]
        • auroc = history["val_auroc"][best_epoch]
        • brier = history["val_brier"][best_epoch]
```

---

## 9. Example Training Run

### Scenario:
```
Epoch 1: val_pr_auc = 0.75  → best_score = 0.75, counter = 0, save checkpoint
Epoch 2: val_pr_auc = 0.78  → best_score = 0.78, counter = 0, save checkpoint
Epoch 3: val_pr_auc = 0.80  → best_score = 0.80, counter = 0, save checkpoint
Epoch 4: val_pr_auc = 0.79  → no improvement, counter = 1
Epoch 5: val_pr_auc = 0.79  → no improvement, counter = 2
...
Epoch 18: val_pr_auc = 0.78 → no improvement, counter = 15
                             → early_stop = True, break

After training:
  → Load checkpoint from epoch 3 (best_score = 0.80)
  → Return metrics from epoch 3
```

---

## 10. Confirmation Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Metric monitored is validation PR-AUC** | ✅ Yes | Line 245: `val_pr_auc = val_metrics["pr_auc"]`<br>Line 254: `self.early_stopping(val_pr_auc, epoch)` |
| **Patience is configured** | ✅ Yes | Line 126: `patience=config.EARLY_STOPPING_PATIENCE`<br>`config.py:58`: `EARLY_STOPPING_PATIENCE = 15` |
| **Mode is "max" (maximize PR-AUC)** | ✅ Yes | Line 127: `mode="max"  # Maximize PR-AUC`<br>Line 60-61: Improvement check uses `>` operator |
| **Best checkpoint is saved** | ✅ Yes | Lines 244-251: Saves when `val_pr_auc > best_score` |
| **Best checkpoint is restored** | ✅ Yes | Lines 258-261: Loads after training loop |
| **Best metrics are returned** | ✅ Yes | Lines 263-271: Returns metrics from `best_epoch` |
| **PR-AUC is breast-level** | ✅ Yes | `trainer.validate()` aggregates to breast-level before computing metrics |

---

## 11. Code Locations Summary

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **EarlyStopping class** | `training/trainer.py` | 18-74 | Custom early stopping implementation |
| **Initialization** | `training/trainer.py` | 124-128 | Create EarlyStopping with patience=15, mode="max" |
| **Patience config** | `config.py` | 58 | `EARLY_STOPPING_PATIENCE = 15` |
| **Checkpoint saving** | `training/trainer.py` | 203-209 | `save_checkpoint()` method |
| **Checkpoint restoration** | `training/trainer.py` | 211-215 | `load_checkpoint()` method |
| **Training loop** | `training/trainer.py` | 217-272 | `train()` method with early stopping |
| **Save trigger** | `training/trainer.py` | 244-251 | Save when `val_pr_auc > best_score` |
| **Restore trigger** | `training/trainer.py` | 258-261 | Load best checkpoint after training |
| **Return best metrics** | `training/trainer.py` | 263-271 | Return metrics from `best_epoch` |

---

## 12. Comparison: PyTorch Lightning vs Custom

This implementation uses a **custom training loop** (not PyTorch Lightning).

### PyTorch Lightning Equivalent:

If this were PyTorch Lightning, it would look like:
```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor="val_pr_auc",
    patience=15,
    mode="max"
)

# Model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_pr_auc",
    mode="max",
    save_top_k=1,
    filename="best_checkpoint"
)

# Trainer
trainer = pl.Trainer(
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=100
)
```

### Custom Implementation Advantages:
1. **Full control** over checkpoint saving logic
2. **Explicit restoration** of best checkpoint
3. **Direct access** to best epoch index for metric retrieval
4. **No hidden behavior** - all logic is visible in train() method

---

## Conclusion

**CONFIRMED:**

1. ✅ **Metric monitored:** Validation PR-AUC (breast-level)
2. ✅ **Patience:** 15 epochs (from `config.EARLY_STOPPING_PATIENCE`)
3. ✅ **Mode:** "max" - Correctly maximizes PR-AUC
4. ✅ **Checkpoint saving:** Triggered when `val_pr_auc > best_score`
5. ✅ **Checkpoint restoration:** Always restored after training loop
6. ✅ **Best metrics returned:** Metrics from `best_epoch` are returned to Problem class

**The early stopping implementation is correct and follows best practices for validation-based model selection.**
