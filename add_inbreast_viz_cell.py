"""
Create INbreast visualization cell for zero_shot_evaluation notebook.
Shows sample mammography images with metadata.
"""

cell_markdown = {
    "cell_type": "markdown",
    "id": "cell-visualize-inbreast-header",
    "metadata": {},
    "source": [
        "## 4a. Visualize INbreast Dataset (Optional)\n\n",
        "Visualize sample INbreast images to verify data loaded correctly.\n\n",
        "**Note:** INbreast images should be in PNG format (converted from DICOM).\n\n",
        "This shows:\n",
        "- Sample malignant and benign cases\n",
        "- Image metadata (view, laterality, BI-RADS)\n",
        "- Dataset statistics"
    ]
}

cell_code = {
    "cell_type": "code",
    "id": "cell-visualize-inbreast",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "from pathlib import Path\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"INBREAST DATASET VISUALIZATION\")\n",
        "print(\"=\" * 80)\n",
        "print()\n",
        "\n",
        "# Select sample images (4 malignant, 4 benign if available)\n",
        "malignant_samples = inbreast_metadata[inbreast_metadata['label'] == 1].head(4)\n",
        "benign_samples = inbreast_metadata[inbreast_metadata['label'] == 0].head(4)\n",
        "\n",
        "# Combine for visualization\n",
        "import pandas as pd\n",
        "samples = pd.concat([malignant_samples, benign_samples])\n",
        "\n",
        "print(f\"Showing {len(samples)} sample images:\")\n",
        "print(f\"  Malignant: {len(malignant_samples)}\")\n",
        "print(f\"  Benign: {len(benign_samples)}\")\n",
        "print()\n",
        "\n",
        "# Create figure\n",
        "n_images = len(samples)\n",
        "n_cols = min(4, n_images)\n",
        "n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division\n",
        "\n",
        "fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))\n",
        "if n_rows == 1 and n_cols == 1:\n",
        "    axes = np.array([axes])\n",
        "elif n_rows == 1 or n_cols == 1:\n",
        "    axes = axes.flatten()\n",
        "else:\n",
        "    axes = axes.flatten()\n",
        "\n",
        "fig.suptitle('Sample INbreast Mammography Images', fontsize=16, fontweight='bold')\n",
        "\n",
        "inbreast_image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG[\"image_dir\"]\n",
        "\n",
        "for idx, (ax, (_, row)) in enumerate(zip(axes, samples.iterrows())):\n",
        "    # Construct image path\n",
        "    if 'image_path' in row and pd.notna(row['image_path']):\n",
        "        img_path = inbreast_image_dir / row['image_path']\n",
        "    else:\n",
        "        # Try common naming patterns\n",
        "        img_id = row['image_id']\n",
        "        # Try different extensions\n",
        "        for ext in ['.png', '.PNG', '.dcm', '.DCM']:\n",
        "            potential_path = inbreast_image_dir / f\"{img_id}{ext}\"\n",
        "            if potential_path.exists():\n",
        "                img_path = potential_path\n",
        "                break\n",
        "    \n",
        "    if img_path.exists():\n",
        "        try:\n",
        "            # Load image\n",
        "            img = Image.open(img_path)\n",
        "            \n",
        "            # Convert to grayscale if needed\n",
        "            if img.mode != 'L':\n",
        "                img = img.convert('L')\n",
        "            \n",
        "            # Display image\n",
        "            ax.imshow(img, cmap='gray')\n",
        "            \n",
        "            # Create title with metadata\n",
        "            label_text = 'Malignant' if row['label'] == 1 else 'Benign'\n",
        "            color = 'red' if row['label'] == 1 else 'green'\n",
        "            \n",
        "            title = f\"{label_text}\\n\"\n",
        "            if 'view' in row and pd.notna(row['view']):\n",
        "                title += f\"View: {row['view']}\"\n",
        "            if 'laterality' in row and pd.notna(row['laterality']):\n",
        "                title += f\" | Lat: {row['laterality']}\"\n",
        "            title += \"\\n\"\n",
        "            if 'birads_original' in row and pd.notna(row['birads_original']):\n",
        "                title += f\"BI-RADS: {row['birads_original']}\"\n",
        "            \n",
        "            ax.set_title(title, fontsize=10, fontweight='bold', color=color)\n",
        "            ax.axis('off')\n",
        "            \n",
        "        except Exception as e:\n",
        "            ax.text(0.5, 0.5, f'Error loading:\\n{img_path.name}\\n{str(e)}', \n",
        "                    ha='center', va='center', fontsize=8, wrap=True)\n",
        "            ax.axis('off')\n",
        "    else:\n",
        "        ax.text(0.5, 0.5, f'Image not found:\\n{row[\"image_id\"]}', \n",
        "                ha='center', va='center', fontsize=10)\n",
        "        ax.axis('off')\n",
        "\n",
        "# Hide extra subplots if any\n",
        "for idx in range(len(samples), len(axes)):\n",
        "    axes[idx].axis('off')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Print dataset statistics\n",
        "print(\"=\" * 80)\n",
        "print(\"INBREAST DATASET SUMMARY\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Total images: {len(inbreast_metadata)}\")\n",
        "print(f\"Patients: {inbreast_metadata['patient_id'].nunique()}\")\n",
        "print(f\"Breasts: {inbreast_metadata['breast_id'].nunique()}\")\n",
        "print()\n",
        "print(\"Label distribution:\")\n",
        "print(f\"  Malignant: {(inbreast_metadata['label'] == 1).sum()} ({(inbreast_metadata['label'] == 1).sum() / len(inbreast_metadata) * 100:.1f}%)\")\n",
        "print(f\"  Benign: {(inbreast_metadata['label'] == 0).sum()} ({(inbreast_metadata['label'] == 0).sum() / len(inbreast_metadata) * 100:.1f}%)\")\n",
        "print()\n",
        "if 'view' in inbreast_metadata.columns:\n",
        "    print(\"View distribution:\")\n",
        "    print(inbreast_metadata['view'].value_counts())\n",
        "    print()\n",
        "if 'laterality' in inbreast_metadata.columns:\n",
        "    print(\"Laterality distribution:\")\n",
        "    print(inbreast_metadata['laterality'].value_counts())\n",
        "    print()\n",
        "if 'birads_original' in inbreast_metadata.columns:\n",
        "    print(\"BI-RADS distribution:\")\n",
        "    print(inbreast_metadata['birads_original'].value_counts())\n",
        "print(\"=\" * 80)\n",
        "\n",
        "# Check image format\n",
        "print()\n",
        "print(\"Image Format Check:\")\n",
        "sample_img_path = inbreast_image_dir / inbreast_metadata.iloc[0]['image_path'] if 'image_path' in inbreast_metadata.columns else None\n",
        "if sample_img_path and sample_img_path.exists():\n",
        "    print(f\"  Sample path: {sample_img_path}\")\n",
        "    print(f\"  Extension: {sample_img_path.suffix}\")\n",
        "    if sample_img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:\n",
        "        print(\"  ✓ Images are in standard format (PNG/JPEG)\")\n",
        "    elif sample_img_path.suffix.lower() in ['.dcm', '.dicom']:\n",
        "        print(\"  ⚠ Images are in DICOM format\")\n",
        "        print(\"  Note: This notebook expects PNG format.\")\n",
        "        print(\"  Please convert DICOM to PNG first using:\")\n",
        "        print(\"    - pydicom + PIL\")\n",
        "        print(\"    - SimpleITK\")\n",
        "        print(\"    - Or use the conversion script in data/preprocessing/\")\n",
        "else:\n",
        "    print(\"  Could not determine image format\")\n",
        "    print(f\"  Image directory: {inbreast_image_dir}\")\n",
        "print(\"=\" * 80)\n"
    ]
}

# Print the cells for manual insertion
print("=" * 80)
print("INbreast VISUALIZATION CELL")
print("=" * 80)
print()
print("Add these cells to zero_shot_evaluation.ipynb after Section 4")
print()
print("=" * 80)
print("MARKDOWN CELL:")
print("=" * 80)
print(''.join(cell_markdown['source']))
print()
print("=" * 80)
print("CODE CELL:")
print("=" * 80)
print(''.join(cell_code['source']))
print()
print("=" * 80)
print()
print("To add programmatically, update zero_shot_evaluation.ipynb:")
print("Insert after cell ID 'cell-load-inbreast'")
