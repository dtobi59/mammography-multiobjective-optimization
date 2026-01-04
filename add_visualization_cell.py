"""
Add dataset visualization cell to Colab notebook.
"""

import json

# Read the notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create visualization cell
viz_cell = {
    "cell_type": "code",
    "id": "cell-visualize-data",
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
        "\n",
        "# Select sample images (2 malignant, 2 benign, different views)\n",
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

# Add markdown cell before visualization
viz_header_cell = {
    "cell_type": "markdown",
    "id": "cell-viz-header",
    "metadata": {},
    "source": [
        "### 4a. Visualize Sample Images\n\n",
        "Let's visualize some sample mammography images to verify the data loaded correctly.\n\n",
        "This shows:\n",
        "- **Top row**: Malignant cases (BI-RADS 4, 5, or 6)\n",
        "- **Bottom row**: Benign cases (BI-RADS 2)\n",
        "- **Metadata**: View type (CC/MLO), laterality (L/R), BI-RADS category"
    ]
}

# Find where to insert (after cell-18, which loads the data)
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('id') == 'cell-18':
        insert_index = i + 1
        break

if insert_index:
    # Insert header and visualization cells
    notebook['cells'].insert(insert_index, viz_header_cell)
    notebook['cells'].insert(insert_index + 1, viz_cell)

    # Save updated notebook
    with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("[OK] Added dataset visualization cell to colab_tutorial.ipynb")
    print(f"     Inserted at position {insert_index} (after Section 4 data loading)")
    print()
    print("New cells added:")
    print("  1. Markdown header: '4a. Visualize Sample Images'")
    print("  2. Code cell: Visualization with matplotlib")
    print()
    print("Visualization shows:")
    print("  - 8 sample images (4 malignant, 4 benign)")
    print("  - 2x4 grid with proper labels and metadata")
    print("  - Dataset statistics (distribution, views, laterality)")
else:
    print("[ERROR] Could not find cell-18 to insert visualization")
