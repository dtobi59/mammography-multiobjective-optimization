"""
Create Colab cell for zero-shot INbreast evaluation.
"""

cell_code = """
# ============================================================================
# ZERO-SHOT EVALUATION ON INBREAST (TARGET DATASET)
# ============================================================================

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import config
from models.resnet import ResNet50WithPartialFineTuning
from data.dataset import create_dataloaders
from training.metrics import compute_metrics
from utils.noisy_or import aggregate_to_breast_level

print("=" * 80)
print("ZERO-SHOT TRANSFER EVALUATION ON INBREAST")
print("=" * 80)
print()

# ============================================================================
# STEP 1: SELECT SOLUTION TO EVALUATE
# ============================================================================

print("Available solutions from Pareto front:")
print(f"  Total solutions: {len(pareto_df)}")
print()
print("Best solutions for each objective:")
print(f"  Best PR-AUC:    Solution {pareto_df['pr_auc'].idxmax()} (PR-AUC: {pareto_df['pr_auc'].max():.4f})")
print(f"  Best AUROC:     Solution {pareto_df['auroc'].idxmax()} (AUROC: {pareto_df['auroc'].max():.4f})")
print(f"  Best Brier:     Solution {pareto_df['brier'].idxmin()} (Brier: {pareto_df['brier'].min():.4f})")
print(f"  Best Robustness: Solution {pareto_df['robustness_degradation'].idxmin()} (Robustness: {pareto_df['robustness_degradation'].min():.4f})")
print()

# Choose which solution to evaluate (default: best PR-AUC)
solution_id = pareto_df['pr_auc'].idxmax()  # Change this to select different solution
print(f"[Selected] Evaluating Solution {solution_id} (Best PR-AUC)")
print()

# Get hyperparameters for selected solution
selected_solution = pareto_df.iloc[solution_id]
print("Hyperparameters:")
print(f"  Learning rate:          {selected_solution['learning_rate']:.6f}")
print(f"  Weight decay:           {selected_solution['weight_decay']:.6f}")
print(f"  Dropout rate:           {selected_solution['dropout_rate']:.4f}")
print(f"  Augmentation strength:  {selected_solution['augmentation_strength']:.4f}")
print(f"  Unfreeze fraction:      {selected_solution['unfreeze_fraction']:.4f}")
print()
print("VinDr (source) performance:")
print(f"  PR-AUC:     {selected_solution['pr_auc']:.4f}")
print(f"  AUROC:      {selected_solution['auroc']:.4f}")
print(f"  Brier:      {selected_solution['brier']:.4f}")
print(f"  Robustness: {selected_solution['robustness_degradation']:.4f}")
print("=" * 80)
print()

# ============================================================================
# STEP 2: LOAD MODEL CHECKPOINT
# ============================================================================

# Find checkpoint directory for this evaluation
checkpoint_path = Path(CHECKPOINT_DIR) / f"eval_{solution_id}" / "best_checkpoint.pt"

if not checkpoint_path.exists():
    print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
    print()
    print("Available checkpoint directories:")
    checkpoint_parent = Path(CHECKPOINT_DIR)
    for d in sorted(checkpoint_parent.glob("eval_*")):
        print(f"  {d.name}")
    raise FileNotFoundError(f"Checkpoint not found for solution {solution_id}")

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"[OK] Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
print()

# ============================================================================
# STEP 3: CREATE MODEL WITH SELECTED HYPERPARAMETERS
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print()

model = ResNet50WithPartialFineTuning(
    num_classes=1,
    dropout_rate=selected_solution['dropout_rate'],
    unfreeze_fraction=selected_solution['unfreeze_fraction'],
)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("[OK] Model created and weights loaded")
print()

# ============================================================================
# STEP 4: LOAD INBREAST DATASET
# ============================================================================

print("=" * 80)
print("LOADING INBREAST DATASET (TARGET)")
print("=" * 80)

# Load INbreast metadata (already loaded in Section 4)
print(f"INbreast images: {len(inbreast_metadata_full)}")
print(f"Patients: {inbreast_metadata_full['patient_id'].nunique()}")
print(f"Breasts: {inbreast_metadata_full['breast_id'].nunique()}")
print()
print("Label distribution:")
print(inbreast_metadata_full['label'].value_counts())
print()

# Create dataloader for INbreast
inbreast_image_dir = str(Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"])

_, inbreast_loader = create_dataloaders(
    train_metadata=inbreast_metadata_full.head(0),  # Empty train set
    val_metadata=inbreast_metadata_full,            # Use all INbreast for eval
    image_dir=inbreast_image_dir,
    batch_size=config.BATCH_SIZE,
    augmentation_strength=0.0,  # No augmentation for evaluation
    num_workers=config.NUM_WORKERS,
)

print(f"[OK] Created dataloader ({len(inbreast_loader)} batches)")
print("=" * 80)
print()

# ============================================================================
# STEP 5: RUN INFERENCE ON INBREAST
# ============================================================================

print("=" * 80)
print("RUNNING INFERENCE ON INBREAST")
print("=" * 80)

all_preds = []
all_labels = []
all_image_ids = []

with torch.no_grad():
    for batch_idx, (images, labels, metadata) in enumerate(inbreast_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze()

        # Store predictions
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_image_ids.extend(metadata['image_id'])

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(inbreast_loader)} batches", end='\\r')

print(f"\\n[OK] Inference complete: {len(all_preds)} images")
print()

# ============================================================================
# STEP 6: IMAGE-LEVEL EVALUATION
# ============================================================================

print("=" * 80)
print("IMAGE-LEVEL EVALUATION")
print("=" * 80)

all_preds_array = np.array(all_preds)
all_labels_array = np.array(all_labels)

image_metrics = compute_metrics(all_preds_array, all_labels_array)

print(f"PR-AUC:        {image_metrics['pr_auc']:.4f}")
print(f"AUROC:         {image_metrics['auroc']:.4f}")
print(f"Brier Score:   {image_metrics['brier']:.4f}")
print(f"Brier Null:    {image_metrics['brier_null']:.4f}")
print(f"Scaled Brier:  {image_metrics['scaled_brier']:.4f}")
print("=" * 80)
print()

# ============================================================================
# STEP 7: BREAST-LEVEL EVALUATION (Noisy OR Aggregation)
# ============================================================================

print("=" * 80)
print("BREAST-LEVEL EVALUATION (NOISY OR AGGREGATION)")
print("=" * 80)

# Create image predictions dictionary
image_predictions = {img_id: pred for img_id, pred in zip(all_image_ids, all_preds)}

# Aggregate to breast level using Noisy OR
breast_preds, breast_labels = aggregate_to_breast_level(
    image_predictions=image_predictions,
    metadata=inbreast_metadata_full
)

print(f"Aggregated to {len(breast_preds)} breasts")
print()

# Compute breast-level metrics
breast_metrics = compute_metrics(breast_preds, breast_labels)

print(f"PR-AUC:        {breast_metrics['pr_auc']:.4f}")
print(f"AUROC:         {breast_metrics['auroc']:.4f}")
print(f"Brier Score:   {breast_metrics['brier']:.4f}")
print(f"Brier Null:    {breast_metrics['brier_null']:.4f}")
print(f"Scaled Brier:  {breast_metrics['scaled_brier']:.4f}")
print("=" * 80)
print()

# ============================================================================
# STEP 8: SAVE EVALUATION RESULTS
# ============================================================================

print("=" * 80)
print("SAVING EVALUATION RESULTS")
print("=" * 80)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create results directory
eval_results_dir = Path(OUTPUT_DIR) / "inbreast_evaluation"
eval_results_dir.mkdir(exist_ok=True)

# Save evaluation results
eval_results = {
    "timestamp": timestamp,
    "solution_id": int(solution_id),
    "hyperparameters": {
        "learning_rate": float(selected_solution['learning_rate']),
        "weight_decay": float(selected_solution['weight_decay']),
        "dropout_rate": float(selected_solution['dropout_rate']),
        "augmentation_strength": float(selected_solution['augmentation_strength']),
        "unfreeze_fraction": float(selected_solution['unfreeze_fraction']),
    },
    "vindr_performance": {
        "pr_auc": float(selected_solution['pr_auc']),
        "auroc": float(selected_solution['auroc']),
        "brier": float(selected_solution['brier']),
        "robustness_degradation": float(selected_solution['robustness_degradation']),
    },
    "inbreast_image_level": {
        "n_images": len(all_preds),
        "pr_auc": float(image_metrics['pr_auc']),
        "auroc": float(image_metrics['auroc']),
        "brier": float(image_metrics['brier']),
        "brier_null": float(image_metrics['brier_null']),
        "scaled_brier": float(image_metrics['scaled_brier']),
    },
    "inbreast_breast_level": {
        "n_breasts": len(breast_preds),
        "pr_auc": float(breast_metrics['pr_auc']),
        "auroc": float(breast_metrics['auroc']),
        "brier": float(breast_metrics['brier']),
        "brier_null": float(breast_metrics['brier_null']),
        "scaled_brier": float(breast_metrics['scaled_brier']),
    },
}

# Save as JSON
import json
results_path = eval_results_dir / f"evaluation_solution_{solution_id}_{timestamp}.json"
with open(results_path, 'w') as f:
    json.dump(eval_results, f, indent=2)

print(f"[OK] Saved results: {results_path}")
print()

# Save predictions as CSV
predictions_df = pd.DataFrame({
    'image_id': all_image_ids,
    'prediction': all_preds,
    'label': all_labels,
})
pred_path = eval_results_dir / f"predictions_solution_{solution_id}_{timestamp}.csv"
predictions_df.to_csv(pred_path, index=False)
print(f"[OK] Saved predictions: {pred_path}")
print()

print("=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  Model: Solution {solution_id} (Best PR-AUC from VinDr)")
print()
print("  VinDr (Source) Performance:")
print(f"    PR-AUC: {selected_solution['pr_auc']:.4f}")
print(f"    AUROC:  {selected_solution['auroc']:.4f}")
print()
print("  INbreast (Target) Performance:")
print(f"    Image-level PR-AUC:  {image_metrics['pr_auc']:.4f}")
print(f"    Image-level AUROC:   {image_metrics['auroc']:.4f}")
print(f"    Breast-level PR-AUC: {breast_metrics['pr_auc']:.4f}")
print(f"    Breast-level AUROC:  {breast_metrics['auroc']:.4f}")
print()
print("Transfer Performance:")
if breast_metrics['pr_auc'] >= selected_solution['pr_auc'] * 0.9:
    print("  ✓ EXCELLENT - Minimal performance degradation!")
elif breast_metrics['pr_auc'] >= selected_solution['pr_auc'] * 0.8:
    print("  ✓ GOOD - Acceptable transfer performance")
else:
    print("  ⚠ MODERATE - Noticeable performance drop (expected for zero-shot)")
print()
print("Results saved to Google Drive:")
print(f"  {eval_results_dir}")
print("=" * 80)
"""

print("=" * 80)
print("INBREAST ZERO-SHOT EVALUATION CELL")
print("=" * 80)
print()
print(cell_code)
print()
print("=" * 80)
print("INSTRUCTIONS:")
print("=" * 80)
print("1. Create a new code cell in Colab (after Section 5a)")
print("2. Copy and paste the code above")
print("3. Run the cell")
print()
print("This will:")
print("  - Load the best model from optimization")
print("  - Evaluate on INbreast (zero-shot, no fine-tuning)")
print("  - Report image-level and breast-level metrics")
print("  - Save results to Google Drive")
print()
print("You can change the solution by modifying:")
print("  solution_id = pareto_df['pr_auc'].idxmax()  # Best PR-AUC")
print("  solution_id = pareto_df['auroc'].idxmax()   # Best AUROC")
print("  solution_id = 5                              # Specific solution")
