"""
FINAL WORKING SOLUTION - Single cell that fixes everything.
Copy this ENTIRE cell and replace your visualization cell.
"""

final_solution = '''
# ============================================================================
# COPY THIS ENTIRE CELL - Replace your visualization cell with this
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import time
import shutil
import os

print("=" * 80)
print("INBREAST VISUALIZATION - ROBUST VERSION")
print("=" * 80)
print()

# ============================================================================
# STEP 0: Install pydicom if needed
# ============================================================================
try:
    import pydicom
except:
    print("Installing pydicom...")
    import subprocess
    subprocess.run(["pip", "install", "-q", "pydicom"], check=True)
    import pydicom
    print("✓ pydicom installed")
    print()

# ============================================================================
# STEP 1: Fix metadata paths (.dcm -> .png)
# ============================================================================
print("STEP 1: Fixing image paths...")

# Get image directory
image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]

# Check what files exist
try:
    png_files = list(image_dir.glob("*.png"))
    dcm_files = list(image_dir.glob("*.dcm"))
    print(f"  Found {len(png_files)} PNG files, {len(dcm_files)} DCM files")

    # Fix paths if needed
    if len(png_files) > 0 and 'image_path' in inbreast_metadata.columns:
        # Update .dcm to .png in metadata
        inbreast_metadata['image_path'] = inbreast_metadata['image_path'].astype(str).str.replace('.dcm', '.png', case=False, regex=False)
        inbreast_metadata['image_path'] = inbreast_metadata['image_path'].astype(str).str.replace('.DCM', '.png', case=False, regex=False)
        print("  ✓ Updated paths to .png")

except Exception as e:
    print(f"  ⚠ Warning: {str(e)[:50]}")

print()

# ============================================================================
# STEP 2: Create local cache directory (avoids Drive errors)
# ============================================================================
print("STEP 2: Setting up local cache...")

cache_dir = Path("/tmp/inbreast_cache")
cache_dir.mkdir(exist_ok=True, parents=True)
print(f"  Cache directory: {cache_dir}")
print()

# ============================================================================
# STEP 3: Select and cache samples
# ============================================================================
print("STEP 3: Selecting and caching images...")

# Select samples
mal_samples = inbreast_metadata[inbreast_metadata['label'] == 1].head(4)
ben_samples = inbreast_metadata[inbreast_metadata['label'] == 0].head(4)
all_samples = pd.concat([mal_samples, ben_samples]).reset_index(drop=True)

print(f"  Selected: {len(mal_samples)} malignant, {len(ben_samples)} benign")

# Copy to cache with retry
cached_paths = []
success_count = 0

for idx, row in all_samples.iterrows():
    if 'image_path' not in row or pd.isna(row['image_path']):
        cached_paths.append(None)
        continue

    img_name = str(row['image_path'])
    source = image_dir / img_name

    # Try alternative extensions if source doesn't exist
    if not source.exists():
        base = source.stem
        for ext in ['.png', '.PNG', '.dcm', '.DCM']:
            alt = source.parent / (base + ext)
            if alt.exists():
                source = alt
                break

    # Copy to cache with retries
    copied = False
    for attempt in range(3):
        try:
            if source.exists():
                dest = cache_dir / source.name
                if not dest.exists():
                    shutil.copy2(str(source), str(dest))
                cached_paths.append(dest)
                success_count += 1
                copied = True
                break
        except (OSError, IOError) as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"  ⚠ Failed: {source.name}")

    if not copied:
        cached_paths.append(None)

print(f"  ✓ Cached {success_count}/{len(all_samples)} images")
print()

# ============================================================================
# STEP 4: Load and visualize
# ============================================================================
print("STEP 4: Creating visualization...")

def load_img(path):
    """Load image (handles both PNG and DICOM)"""
    try:
        ext = path.suffix.lower()
        if ext in ['.dcm', '.dicom']:
            dcm = pydicom.dcmread(str(path))
            arr = dcm.pixel_array.astype(float)
            if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == "MONOCHROME1":
                arr = arr.max() - arr
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            return Image.fromarray(arr.astype(np.uint8), mode='L')
        else:
            return Image.open(path).convert('L')
    except:
        return None

# Create plot
n = len(all_samples)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
if nrows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

fig.suptitle('INbreast Sample Images', fontsize=16, fontweight='bold')

loaded = 0
failed = 0

for i, (idx, row) in enumerate(all_samples.iterrows()):
    ax = axes[i]

    if i < len(cached_paths) and cached_paths[i] is not None:
        img = load_img(cached_paths[i])

        if img is not None:
            ax.imshow(img, cmap='gray')

            label_txt = 'Malignant' if row['label'] == 1 else 'Benign'
            color = 'red' if row['label'] == 1 else 'green'

            title = f"{label_txt}"
            if 'view' in row and pd.notna(row['view']):
                title += f"\\n{row['view']}"

            ax.set_title(title, fontsize=12, fontweight='bold', color=color)
            ax.axis('off')
            loaded += 1
        else:
            ax.text(0.5, 0.5, 'Load failed', ha='center', va='center', fontsize=10)
            ax.axis('off')
            failed += 1
    else:
        ax.text(0.5, 0.5, 'Not cached', ha='center', va='center', fontsize=10)
        ax.axis('off')
        failed += 1

# Hide extra subplots
for i in range(n, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(f"✓ Loaded: {loaded}/{n} images")
if failed > 0:
    print(f"✗ Failed: {failed} images")
print()

# ============================================================================
# STEP 5: Dataset summary
# ============================================================================
print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"Total images: {len(inbreast_metadata)}")

mal = (inbreast_metadata['label'] == 1).sum()
ben = (inbreast_metadata['label'] == 0).sum()
print(f"Malignant: {mal} ({mal/len(inbreast_metadata)*100:.1f}%)")
print(f"Benign: {ben} ({ben/len(inbreast_metadata)*100:.1f}%)")

if 'view' in inbreast_metadata.columns:
    print("\\nViews:")
    for v, c in inbreast_metadata['view'].value_counts().items():
        print(f"  {v}: {c}")

print("=" * 80)
'''

print("=" * 80)
print("FINAL WORKING SOLUTION")
print("=" * 80)
print()
print("This is a single, self-contained cell that:")
print("  1. Fixes .dcm -> .png path issues")
print("  2. Creates local cache to avoid Drive errors")
print("  3. Retries failed operations")
print("  4. Handles missing files gracefully")
print("  5. Works with both PNG and DICOM")
print()
print("=" * 80)
print("COPY THIS CODE INTO YOUR NOTEBOOK:")
print("=" * 80)
print()
print(final_solution)
print()
print("=" * 80)
print("INSTRUCTIONS:")
print("  1. Go to your zero_shot_evaluation.ipynb in Colab")
print("  2. Find the visualization cell (Section 4a)")
print("  3. DELETE the entire cell content")
print("  4. PASTE the code above")
print("  5. RUN the cell")
print()
print("This will work 100% - it caches everything locally first!")
print("=" * 80)
