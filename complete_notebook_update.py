"""
Complete Colab notebook update: Add stratified subsampling + visualization.
"""

import json

# Read the notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# ============================================================================
# 1. CREATE SECTION 4A: STRATIFIED SUBSAMPLING
# ============================================================================

section_4a_markdown = {
    "cell_type": "markdown",
    "id": "cell-4a-header",
    "metadata": {},
    "source": [
        "## 4a. Create Stratified Subsample (Research-Grade Pipeline)\n\n",
        "Create a compute-efficient stratified subsample with research-grade quality:\n",
        "- **1000 images total** (250 malignant, 750 benign)\n",
        "- **Exclude BI-RADS 3** (uncertain cases)\n",
        "- **Patient-wise split FIRST** (prevents data leakage)\n",
        "- **Stratified sampling** within train/val partitions\n",
        "- **Saves manifest** for reproducibility"
    ]
}

section_4a_code = {
    "cell_type": "code",
    "id": "cell-4a-subsample",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# CREATE STRATIFIED SUBSAMPLE FOR COMPUTE-EFFICIENT TRAINING\n",
        "# ============================================================================\n",
        "\n",
        "from data.dataset import create_stratified_subsample\n",
        "import os\n",
        "\n",
        "# Store full metadata for reference\n",
        "vindr_metadata_full = vindr_metadata.copy()\n",
        "inbreast_metadata_full = inbreast_metadata.copy()\n",
        "\n",
        "# Create manifests directory in Google Drive\n",
        "manifests_dir = \"/content/drive/MyDrive/vindr_optimization/manifests\"\n",
        "os.makedirs(manifests_dir, exist_ok=True)\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"CREATING STRATIFIED SUBSAMPLE\")\n",
        "print(\"=\" * 80)\n",
        "\n",
        "# Create stratified subsample\n",
        "train_metadata, val_metadata, manifest_path = create_stratified_subsample(\n",
        "    metadata=vindr_metadata_full,\n",
        "    target_total=1000,          # 1000 images total\n",
        "    target_malignant=250,       # 250 malignant\n",
        "    target_benign=750,          # 750 benign\n",
        "    train_ratio=0.8,            # 80/20 split\n",
        "    random_seed=42,\n",
        "    output_dir=manifests_dir,   # Save to Google Drive\n",
        "    exclude_birads_3=True,      # Exclude BI-RADS 3\n",
        ")\n",
        "\n",
        "print(\"\\n[OK] Stratified subsample created successfully!\")\n",
        "print(f\"[OK] Manifest saved to: {manifest_path}\")\n",
        "print(\"\\nVariables created:\")\n",
        "print(\"  - train_metadata: Training set metadata\")\n",
        "print(\"  - val_metadata: Validation set metadata\")\n",
        "print(\"  - manifest_path: Path to saved manifest CSV\")\n",
        "print(\"  - vindr_metadata_full: Full VinDr dataset (for reference)\")\n",
        "print(\"  - inbreast_metadata_full: Full INbreast dataset (for evaluation)\")\n",
        "print(\"=\" * 80)\n"
    ]
}

# ============================================================================
# 2. CREATE SECTION 4B: VISUALIZATION
# ============================================================================

section_4b_markdown = {
    "cell_type": "markdown",
    "id": "cell-4b-header",
    "metadata": {},
    "source": [
        "## 4b. Visualize Sample Images\n\n",
        "Verify the data loaded correctly by visualizing sample mammography images.\n\n",
        "This shows:\n",
        "- **Top row**: Malignant cases (BI-RADS 4, 5, or 6)\n",
        "- **Bottom row**: Benign cases (BI-RADS 2)\n",
        "- **Metadata**: View type (CC/MLO), laterality (L/R), BI-RADS category"
    ]
}

section_4b_code = {
    "cell_type": "code",
    "id": "cell-4b-viz",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# VISUALIZE SAMPLE IMAGES FROM DATASET\n",
        "# ============================================================================\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "import os\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "\n",
        "# Select sample images (4 malignant, 4 benign, different views)\n",
        "malignant_samples = train_metadata[train_metadata['label'] == 1].head(4)\n",
        "benign_samples = train_metadata[train_metadata['label'] == 0].head(4)\n",
        "\n",
        "# Combine for visualization\n",
        "samples = pd.concat([malignant_samples, benign_samples])\n",
        "\n",
        "# Create figure\n",
        "fig, axes = plt.subplots(2, 4, figsize=(20, 10))\n",
        "fig.suptitle('Sample Mammography Images from Training Set', fontsize=16, fontweight='bold')\n",
        "\n",
        "image_dir = Path(config.VINDR_MAMMO_PATH) / config.VINDR_CONFIG[\"image_dir\"]\n",
        "\n",
        "for idx, (ax, (_, row)) in enumerate(zip(axes.flat, samples.iterrows())):\n",
        "    # Load image\n",
        "    img_path = image_dir / row['image_path']\n",
        "    \n",
        "    if img_path.exists():\n",
        "        img = Image.open(img_path).convert('L')\n",
        "        \n",
        "        # Display image\n",
        "        ax.imshow(img, cmap='gray')\n",
        "        \n",
        "        # Create title with metadata\n",
        "        label_text = 'Malignant' if row['label'] == 1 else 'Benign'\n",
        "        color = 'red' if row['label'] == 1 else 'green'\n",
        "        \n",
        "        title = f\"{label_text}\\n\"\n",
        "        title += f\"View: {row['view']} | Laterality: {row.get('laterality', 'N/A')}\\n\"\n",
        "        if 'birads_original' in row:\n",
        "            title += f\"BI-RADS: {row['birads_original']}\"\n",
        "        \n",
        "        ax.set_title(title, fontsize=10, fontweight='bold', color=color)\n",
        "        ax.axis('off')\n",
        "    else:\n",
        "        ax.text(0.5, 0.5, f'Image not found:\\n{row[\"image_id\"]}', \n",
        "                ha='center', va='center', fontsize=10)\n",
        "        ax.axis('off')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Print statistics\n",
        "print(\"\\n\" + \"=\" * 80)\n",
        "print(\"DATASET VISUALIZATION SUMMARY\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Showing {len(samples)} sample images:\")\n",
        "print(f\"  - Top row: Malignant cases (BI-RADS 4, 5, 6)\")\n",
        "print(f\"  - Bottom row: Benign cases (BI-RADS 2)\")\n",
        "print()\n",
        "print(f\"Training set distribution:\")\n",
        "print(f\"  Total images: {len(train_metadata)}\")\n",
        "print(f\"  Malignant: {(train_metadata['label'] == 1).sum()} ({(train_metadata['label'] == 1).sum() / len(train_metadata) * 100:.1f}%)\")\n",
        "print(f\"  Benign: {(train_metadata['label'] == 0).sum()} ({(train_metadata['label'] == 0).sum() / len(train_metadata) * 100:.1f}%)\")\n",
        "print()\n",
        "print(f\"View distribution:\")\n",
        "print(train_metadata['view'].value_counts())\n",
        "print()\n",
        "if 'laterality' in train_metadata.columns:\n",
        "    print(f\"Laterality distribution:\")\n",
        "    print(train_metadata['laterality'].value_counts())\n",
        "    print()\n",
        "print(\"=\" * 80)\n"
    ]
}

# ============================================================================
# 3. INSERT CELLS INTO NOTEBOOK
# ============================================================================

# Find insertion point (after cell 18)
insert_index = 19  # Right after cell 18, before current cell 19 (Section 5)

# Insert all 4 new cells
cells_to_insert = [
    section_4a_markdown,
    section_4a_code,
    section_4b_markdown,
    section_4b_code
]

for i, cell in enumerate(cells_to_insert):
    notebook['cells'].insert(insert_index + i, cell)

# Save updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("[OK] Updated colab_tutorial.ipynb successfully!")
print()
print("Added 4 new cells after cell 18:")
print("  1. Section 4a markdown: 'Create Stratified Subsample'")
print("  2. Section 4a code: Stratified subsampling with manifest saving")
print("  3. Section 4b markdown: 'Visualize Sample Images'")
print("  4. Section 4b code: Image visualization with matplotlib")
print()
print("New sections:")
print("  - Section 4a: Creates train_metadata, val_metadata, manifest_path")
print("  - Section 4b: Visualizes 8 sample images (4 malignant, 4 benign)")
print()
print("Variables created:")
print("  - train_metadata: 800 images (stratified)")
print("  - val_metadata: 200 images (stratified)")
print("  - manifest_path: Path to saved manifest CSV")
print("  - vindr_metadata_full: Full VinDr dataset")
print("  - inbreast_metadata_full: Full INbreast dataset")
