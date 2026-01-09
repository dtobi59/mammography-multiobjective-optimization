"""
Update zero_shot_evaluation.ipynb with working visualization and DICOM conversion cells.
"""

import json
from pathlib import Path

# Read notebook
notebook_path = Path("zero_shot_evaluation.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# ============================================================================
# IMPROVED DICOM TO PNG CONVERSION CELL
# ============================================================================
improved_dicom_conversion = """import pydicom
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

print("=" * 80)
print("DICOM TO PNG CONVERSION (ROBUST VERSION)")
print("=" * 80)
print()

# Set paths - UPDATE THESE IF YOUR STRUCTURE IS DIFFERENT
dicom_dir = Path(INBREAST_PATH) / "AllDICOMs"  # Or use "images" if that's where your DICOMs are
png_output_dir = Path(INBREAST_PATH) / "images_png"

# Create output directory
png_output_dir.mkdir(exist_ok=True)

print(f"Input (DICOM):  {dicom_dir}")
print(f"Output (PNG):   {png_output_dir}")
print()

# Find all DICOM files
dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
print(f"Found {len(dicom_files)} DICOM files")
print()

if len(dicom_files) == 0:
    print("⚠ No DICOM files found!")
    print(f"  Checked: {dicom_dir}")
    print("  Please update the dicom_dir path above")
else:
    print("Converting DICOM to PNG with retry logic...")
    print()

    conversion_errors = []
    success_count = 0

    for dicom_path in tqdm(dicom_files, desc="Converting"):
        converted = False

        # Try up to 3 times per file (handles Drive disconnections)
        for attempt in range(3):
            try:
                # Read DICOM
                dcm = pydicom.dcmread(str(dicom_path))
                img_array = dcm.pixel_array.astype(float)

                # Apply PhotometricInterpretation if needed
                if hasattr(dcm, 'PhotometricInterpretation'):
                    if dcm.PhotometricInterpretation == "MONOCHROME1":
                        # Invert for MONOCHROME1 (lower values = brighter)
                        img_array = img_array.max() - img_array

                # Normalize to 0-255
                img_min, img_max = img_array.min(), img_array.max()
                if img_max > img_min:
                    img_array = (img_array - img_min) / (img_max - img_min) * 255.0
                else:
                    img_array = np.zeros_like(img_array)

                img_array = img_array.astype(np.uint8)

                # Convert to PIL Image
                img = Image.fromarray(img_array, mode='L')

                # Save as PNG with same filename
                png_filename = dicom_path.stem + ".png"
                png_path = png_output_dir / png_filename

                # Save with retry (handles temporary Drive issues)
                img.save(str(png_path), "PNG")

                success_count += 1
                converted = True
                break  # Success - exit retry loop

            except (OSError, IOError) as e:
                # Handle Drive connection errors
                if "107" in str(e) or "Transport endpoint" in str(e):
                    if attempt < 2:  # Retry
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        conversion_errors.append((dicom_path.name, "Drive connection timeout"))
                else:
                    # Other error - don't retry
                    conversion_errors.append((dicom_path.name, str(e)[:100]))
                    break

            except Exception as e:
                # Unexpected error
                conversion_errors.append((dicom_path.name, str(e)[:100]))
                break

        if not converted:
            # Failed after all retries
            pass

    print()
    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Total files: {len(dicom_files)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {len(conversion_errors)}")
    print(f"PNG files saved to: {png_output_dir}")
    print()

    if conversion_errors:
        print(f"⚠ Errors: {len(conversion_errors)} files failed")
        print("First 10 errors:")
        for filename, error in conversion_errors[:10]:
            print(f"  {filename}: {error}")
        print()

        if len(conversion_errors) < len(dicom_files) * 0.1:  # Less than 10% failed
            print("✓ Most files converted successfully (>90%)")
            print("  You can proceed with the available images")
        else:
            print("⚠ WARNING: Many files failed to convert")
            print("  This might indicate a Google Drive connection issue")
            print("  Consider remounting Drive and trying again")
    else:
        print("✓ All files converted successfully!")

    print()
    print("⚠ IMPORTANT: Update image directory in config!")
    print(f"  Change 'image_dir' from '{config.INBREAST_CONFIG['image_dir']}' to 'images_png'")
    print("  Run the configuration cell below to update it")
    print("=" * 80)
"""

# ============================================================================
# IMPROVED VISUALIZATION CELL
# ============================================================================
improved_visualization = """import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import time
import shutil
import os

print("=" * 80)
print("INBREAST DATASET VISUALIZATION (ROBUST VERSION)")
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
# STEP 1: Fix metadata paths (.dcm -> .png if needed)
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
    \"\"\"Load image (handles both PNG and DICOM)\"\"\"
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
    print(f"⚠ Failed: {failed} images")
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
"""

# Update cells in notebook
for cell in notebook['cells']:
    cell_id = cell.get('id', '')

    # Update DICOM conversion cell
    if cell_id == 't1m5t11dana':
        print(f"Updating cell: {cell_id} (DICOM conversion)")
        cell['source'] = improved_dicom_conversion

    # Update visualization cell
    elif cell_id == 'cell-visualize-inbreast':
        print(f"Updating cell: {cell_id} (Visualization)")
        cell['source'] = improved_visualization

# Write updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print()
print("=" * 80)
print("✓ Notebook updated successfully!")
print("=" * 80)
print()
print("Updated cells:")
print("  1. t1m5t11dana: DICOM to PNG conversion (with retry logic)")
print("  2. cell-visualize-inbreast: Visualization (with local caching)")
print()
print("Changes:")
print("  • DICOM conversion now retries failed files 3 times")
print("  • Handles Google Drive connection timeouts (errno 107)")
print("  • Visualization uses local cache to eliminate Drive errors")
print("  • Auto-fixes .dcm to .png path mismatches")
print("  • Comprehensive error reporting")
print("=" * 80)
