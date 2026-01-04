"""
Fix dataset verification cell to check for variables BEFORE using them.
"""

import json

# Read the notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find cell-5a-dataset
dataset_cell_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('id') == 'cell-5a-dataset':
        dataset_cell_index = i
        break

if dataset_cell_index is None:
    print("[ERROR] Could not find cell-5a-dataset")
    exit(1)

print(f"Found dataset verification cell at index {dataset_cell_index}")

# Create corrected cell with check FIRST
corrected_cell = {
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
        "# ============================================================================\n",
        "# CHECK FOR STRATIFIED DATA FROM SECTION 4a - DO THIS FIRST!\n",
        "# ============================================================================\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"VERIFYING STRATIFIED SUBSAMPLE FROM SECTION 4a\")\n",
        "print(\"=\" * 80)\n",
        "\n",
        "# Check if variables exist BEFORE trying to use them\n",
        "try:\n",
        "    # Try to access the variables\n",
        "    _ = train_metadata\n",
        "    _ = val_metadata\n",
        "    _ = manifest_path\n",
        "    print(\"[OK] Found stratified subsample variables\")\n",
        "except NameError:\n",
        "    print()\n",
        "    print(\"[ERROR] Required variables not found!\")\n",
        "    print()\n",
        "    print(\"Please run Section 4a first to create the stratified subsample.\")\n",
        "    print()\n",
        "    print(\"Section 4a creates:\")\n",
        "    print(\"  - train_metadata: Training set (800 images)\")\n",
        "    print(\"  - val_metadata: Validation set (200 images)\")\n",
        "    print(\"  - manifest_path: Path to saved manifest CSV\")\n",
        "    print()\n",
        "    print(\"Go back and run Section 4a, then run this cell again.\")\n",
        "    print(\"=\" * 80)\n",
        "    raise RuntimeError(\"Missing stratified subsample. Run Section 4a first.\")\n",
        "\n",
        "# Now we can safely use the variables\n",
        "print()\n",
        "print(f\"Train: {len(train_metadata)} images, {train_metadata['breast_id'].nunique()} breasts\")\n",
        "print(f\"  Malignant: {(train_metadata['label'] == 1).sum()}\")\n",
        "print(f\"  Benign: {(train_metadata['label'] == 0).sum()}\")\n",
        "print()\n",
        "print(f\"Val: {len(val_metadata)} images, {val_metadata['breast_id'].nunique()} breasts\")\n",
        "print(f\"  Malignant: {(val_metadata['label'] == 1).sum()}\")\n",
        "print(f\"  Benign: {(val_metadata['label'] == 0).sum()}\")\n",
        "print()\n",
        "print(f\"Manifest: {manifest_path}\")\n",
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

# Replace the cell
notebook['cells'][dataset_cell_index] = corrected_cell

# Save updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print()
print("[OK] Fixed dataset verification cell!")
print()
print("Changes:")
print("  - Moved variable existence check to the beginning")
print("  - Used try/except to catch NameError before accessing variables")
print("  - Clear error message tells user to run Section 4a first")
print()
print("Now when users run Section 5a without running Section 4a:")
print("  - They get a clear error message")
print("  - Instructions to go back and run Section 4a")
print("  - No confusing NameError traceback")
