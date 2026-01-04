"""
Split Section 5a into two cells: Dataset verification + Optimization.
"""

import json

# Read the notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell-opt cells (should be around index 23-24 now after insertions)
opt_cell_indices = []
for i, cell in enumerate(notebook['cells']):
    if cell.get('id') == 'cell-opt':
        opt_cell_indices.append(i)

print(f"Found {len(opt_cell_indices)} optimization cells at indices: {opt_cell_indices}")

if len(opt_cell_indices) >= 2:
    # Remove duplicate cell-opt (keep only the first one)
    print(f"Removing duplicate cell at index {opt_cell_indices[1]}")
    notebook['cells'].pop(opt_cell_indices[1])

# Find the remaining cell-opt
opt_cell_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('id') == 'cell-opt':
        opt_cell_index = i
        break

if opt_cell_index is None:
    print("[ERROR] Could not find cell-opt")
    exit(1)

print(f"Working with cell at index {opt_cell_index}")

# ============================================================================
# CREATE TWO NEW CELLS TO REPLACE THE SINGLE cell-opt
# ============================================================================

# Cell 1: Dataset Verification and A100 Setup
dataset_setup_cell = {
    "cell_type": "code",
    "id": "cell-5a-dataset",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# A100 GPU OPTIMIZATIONS AND DATASET VERIFICATION\n",
        "# ============================================================================\n",
        "\n",
        "import torch\n",
        "import config\n",
        "from optimization.nsga3_runner import NSGA3Runner\n",
        "from pathlib import Path\n",
        "\n",
        "# Apply A100 GPU optimizations\n",
        "print(\"=\" * 80)\n",
        "print(\"A100 80GB GPU OPTIMIZATIONS\")\n",
        "print(\"=\" * 80)\n",
        "\n",
        "# Enable TF32 for faster matrix operations on A100\n",
        "torch.backends.cuda.matmul.allow_tf32 = True\n",
        "torch.backends.cudnn.allow_tf32 = True\n",
        "print(\"[OK] TF32 enabled for tensor cores\")\n",
        "\n",
        "# Verify batch size is optimized\n",
        "print(f\"[OK] Batch size: {config.BATCH_SIZE}\")\n",
        "print(f\"[OK] Expected GPU memory: ~{config.BATCH_SIZE * 0.3:.1f} GB / 80 GB\")\n",
        "print(f\"[OK] Speedup vs batch_size=16: ~{config.BATCH_SIZE / 16:.1f}x faster\")\n",
        "\n",
        "# Optional: Scale learning rate with batch size\n",
        "if config.BATCH_SIZE != 16:\n",
        "    lr_scale = (config.BATCH_SIZE / 16) ** 0.5  # sqrt scaling\n",
        "    old_lr = config.HYPERPARAMETER_BOUNDS[\"learning_rate\"]\n",
        "    config.HYPERPARAMETER_BOUNDS[\"learning_rate\"] = (\n",
        "        old_lr[0] * lr_scale,\n",
        "        old_lr[1] * lr_scale\n",
        "    )\n",
        "    print(f\"[OK] Learning rate scaled by {lr_scale:.3f}x for larger batches\")\n",
        "    print(f\"    New range: ({config.HYPERPARAMETER_BOUNDS['learning_rate'][0]:.2e}, \"\n",
        "          f\"{config.HYPERPARAMETER_BOUNDS['learning_rate'][1]:.2e})\")\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print()\n",
        "\n",
        "# Verify we have the stratified data from Section 4a\n",
        "print(\"=\" * 80)\n",
        "print(\"VERIFYING STRATIFIED SUBSAMPLE FROM SECTION 4a\")\n",
        "print(\"=\" * 80)\n",
        "\n",
        "if 'train_metadata' not in locals() or 'val_metadata' not in locals():\n",
        "    print(\"[ERROR] train_metadata or val_metadata not found!\")\n",
        "    print(\"Please run Section 4a first to create the stratified subsample.\")\n",
        "    raise RuntimeError(\"Missing stratified subsample. Run Section 4a first.\")\n",
        "\n",
        "print(f\"Train: {len(train_metadata)} images, {train_metadata['breast_id'].nunique()} breasts\")\n",
        "print(f\"  Malignant: {(train_metadata['label'] == 1).sum()}\")\n",
        "print(f\"  Benign: {(train_metadata['label'] == 0).sum()}\")\n",
        "print()\n",
        "print(f\"Val: {len(val_metadata)} images, {val_metadata['breast_id'].nunique()} breasts\")\n",
        "print(f\"  Malignant: {(val_metadata['label'] == 1).sum()}\")\n",
        "print(f\"  Benign: {(val_metadata['label'] == 0).sum()}\")\n",
        "print(\"=\" * 80)\n",
        "print()\n",
        "\n",
        "# Setup image directory\n",
        "image_dir = str(Path(config.VINDR_MAMMO_PATH) / config.VINDR_CONFIG[\"image_dir\"])\n",
        "print(f\"[OK] Image directory: {image_dir}\")\n",
        "print(f\"[OK] Checkpoints: {CHECKPOINT_DIR}\")\n",
        "print(f\"[OK] Results: {OUTPUT_DIR}\")\n",
        "print()\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"READY TO START OPTIMIZATION\")\n",
        "print(\"=\" * 80)\n",
        "print(\"Run the next cell to start the optimization.\")\n",
        "print(\"=\" * 80)\n"
    ]
}

# Cell 2: Run Optimization
optimization_cell = {
    "cell_type": "code",
    "id": "cell-5a-optimize",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# START NSGA-III OPTIMIZATION\n",
        "# ============================================================================\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"STARTING OPTIMIZATION\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Population size: {config.NSGA3_CONFIG['pop_size']}\")\n",
        "print(f\"Generations: {config.NSGA3_CONFIG['n_generations']}\")\n",
        "print(f\"Total models to train: {config.NSGA3_CONFIG['pop_size'] * config.NSGA3_CONFIG['n_generations']}\")\n",
        "print()\n",
        "print(f\"Estimated time per model: ~7 minutes\")\n",
        "print(f\"Estimated time per generation: ~{24 * 7 / 60:.1f} hours\")\n",
        "print(f\"Estimated total time: ~{24 * 50 * 7 / 60 / 24:.1f} days\")\n",
        "print()\n",
        "print(\"Research Pipeline Features:\")\n",
        "print(\"  [OK] Stratified subsample (1000 images, 250/750 malignant/benign)\")\n",
        "print(\"  [OK] Patient-wise split (no data leakage)\")\n",
        "print(\"  [OK] Generalized Noisy OR (handles variable views per breast)\")\n",
        "print(\"  [OK] Scaled Brier score (logged for calibration assessment)\")\n",
        "print(\"  [OK] Deterministic robustness (reproducible perturbations)\")\n",
        "print(\"=\" * 80)\n",
        "print()\n",
        "\n",
        "# Create runner\n",
        "runner = NSGA3Runner(\n",
        "    train_metadata=train_metadata,\n",
        "    val_metadata=val_metadata,\n",
        "    image_dir=image_dir,\n",
        "    output_dir=OUTPUT_DIR,          # Google Drive\n",
        "    checkpoint_dir=CHECKPOINT_DIR,  # Google Drive\n",
        "    save_frequency=1                # Save after every generation\n",
        ")\n",
        "\n",
        "# Run optimization\n",
        "print(\"Starting optimization...\")\n",
        "print(\"All progress will be saved to Google Drive automatically!\")\n",
        "print()\n",
        "\n",
        "result = runner.run()\n",
        "\n",
        "print()\n",
        "print(\"=\" * 80)\n",
        "print(\"OPTIMIZATION COMPLETE!\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Pareto front size: {len(result.F)}\")\n",
        "print(f\"Results saved to: {runner.output_dir}\")\n",
        "print(f\"Manifest saved to: {manifest_path}\")\n",
        "print(\"=\" * 80)\n"
    ]
}

# ============================================================================
# REPLACE THE OLD cell-opt WITH TWO NEW CELLS
# ============================================================================

# Remove the old cell-opt
notebook['cells'].pop(opt_cell_index)

# Insert the two new cells at the same position
notebook['cells'].insert(opt_cell_index, optimization_cell)
notebook['cells'].insert(opt_cell_index, dataset_setup_cell)

# Save updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print()
print("[OK] Split Section 5a into two cells successfully!")
print()
print("Replaced 1 cell with 2 cells:")
print(f"  1. Cell {opt_cell_index}: Dataset verification and A100 setup")
print(f"  2. Cell {opt_cell_index + 1}: Start optimization")
print()
print("Benefits:")
print("  - Users can verify data before starting long optimization")
print("  - Clear separation of setup vs execution")
print("  - Easier to re-run optimization without re-verifying")
