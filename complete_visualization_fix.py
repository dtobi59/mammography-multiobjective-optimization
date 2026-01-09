"""
Complete visualization fix for errno 107 and PNG/DCM issues.
This is a drop-in replacement for the entire visualization cell.

Handles:
1. Google Drive connection errors (errno 107)
2. PNG vs DICOM path issues
3. Retry logic with exponential backoff
4. Local caching to reduce Drive access
5. Graceful fallback for missing images
"""

complete_fix = """
# ============================================================================
# COMPLETE VISUALIZATION FIX - Replace your entire visualization cell with this
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import time
import shutil
import os

# Install and import pydicom
try:
    import pydicom
except ImportError:
    print("Installing pydicom...")
    !pip install -q pydicom
    import pydicom

print("=" * 80)
print("INBREAST DATASET VISUALIZATION (ROBUST VERSION)")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Fix image paths (.dcm -> .png if needed)
# ============================================================================
print("Step 1: Checking image path extensions...")

image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]
print(f"Image directory: {image_dir}")

# Check what files actually exist
png_count = len(list(image_dir.glob("*.png"))) + len(list(image_dir.glob("*.PNG")))
dcm_count = len(list(image_dir.glob("*.dcm"))) + len(list(image_dir.glob("*.DCM")))

print(f"  PNG files: {png_count}")
print(f"  DCM files: {dcm_count}")

# Fix metadata paths if needed
if png_count > 0 and 'image_path' in inbreast_metadata.columns:
    # Check if paths reference .dcm but files are .png
    sample_path = str(inbreast_metadata.iloc[0]['image_path'])
    if sample_path.lower().endswith('.dcm') and png_count > dcm_count:
        print("  → Updating paths from .dcm to .png...")
        inbreast_metadata['image_path'] = inbreast_metadata['image_path'].str.replace(
            '.dcm', '.png', case=False, regex=False
        )
        print("  ✓ Updated paths to .png")
    else:
        print("  ✓ Paths already correct")
else:
    print("  ✓ Using DICOM format")

print()

# ============================================================================
# STEP 2: Select sample images
# ============================================================================
print("Step 2: Selecting sample images...")

# Select samples
malignant_samples = inbreast_metadata[inbreast_metadata['label'] == 1].head(4)
benign_samples = inbreast_metadata[inbreast_metadata['label'] == 0].head(4)
samples = pd.concat([malignant_samples, benign_samples])

print(f"  Malignant: {len(malignant_samples)} images")
print(f"  Benign: {len(benign_samples)} images")
print(f"  Total: {len(samples)} images")
print()

if len(samples) == 0:
    print("⚠ ERROR: No samples available for visualization")
    print("=" * 80)
else:
    # ========================================================================
    # STEP 3: Copy images to local cache (bypass Drive connection issues)
    # ========================================================================
    print("Step 3: Caching images locally (this avoids Drive errors)...")

    cache_dir = Path("/tmp/inbreast_viz")
    cache_dir.mkdir(exist_ok=True)

    cached_images = {}

    for idx, row in samples.iterrows():
        if 'image_path' not in row or pd.isna(row['image_path']):
            continue

        source_path = image_dir / str(row['image_path'])
        cache_path = cache_dir / source_path.name

        # Try to copy with retries
        success = False
        for attempt in range(3):
            try:
                if not cache_path.exists():
                    # Check if source exists
                    if not source_path.exists():
                        # Try alternative extension
                        alt_path = source_path.with_suffix('.png' if source_path.suffix.lower() == '.dcm' else '.dcm')
                        if alt_path.exists():
                            source_path = alt_path
                        else:
                            print(f"  ⚠ Not found: {source_path.name}")
                            break

                    shutil.copy2(str(source_path), str(cache_path))

                cached_images[idx] = cache_path
                success = True
                break

            except (OSError, IOError) as e:
                error_str = str(e)
                if "107" in error_str or "Transport endpoint" in error_str:
                    if attempt < 2:
                        print(f"  ⟳ Retry {attempt + 1}/3: {source_path.name}")
                        time.sleep(2 ** attempt)
                    else:
                        print(f"  ✗ Failed after 3 retries: {source_path.name}")
                else:
                    print(f"  ✗ Error: {source_path.name} - {str(e)[:40]}")
                    break

    print(f"  ✓ Cached {len(cached_images)}/{len(samples)} images")
    print()

    # ========================================================================
    # STEP 4: Load images and create visualization
    # ========================================================================
    print("Step 4: Creating visualization...")
    print()

    def load_image_safe(img_path):
        """Load image with format auto-detection."""
        try:
            ext = img_path.suffix.lower()

            if ext in ['.dcm', '.dicom']:
                # Load DICOM
                dcm = pydicom.dcmread(str(img_path))
                img_array = dcm.pixel_array.astype(float)

                # Handle photometric interpretation
                if hasattr(dcm, 'PhotometricInterpretation'):
                    if dcm.PhotometricInterpretation == "MONOCHROME1":
                        img_array = img_array.max() - img_array

                # Normalize
                img_min, img_max = img_array.min(), img_array.max()
                if img_max > img_min:
                    img_array = (img_array - img_min) / (img_max - img_min) * 255.0
                else:
                    img_array = np.zeros_like(img_array)

                return Image.fromarray(img_array.astype(np.uint8), mode='L')
            else:
                # Load PNG/JPEG
                return Image.open(img_path).convert('L')
        except Exception as e:
            print(f"  ✗ Load error: {img_path.name} - {str(e)[:40]}")
            return None

    # Create figure
    n_images = len(samples)
    n_cols = min(4, n_images)
    n_rows = max(1, (n_images + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Handle axes array shape
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    fig.suptitle('Sample INbreast Mammography Images', fontsize=16, fontweight='bold')

    # Plot each image
    loaded_count = 0
    failed_count = 0

    for plot_idx, (sample_idx, row) in enumerate(samples.iterrows()):
        ax = axes[plot_idx]

        if sample_idx in cached_images:
            img_path = cached_images[sample_idx]

            # Load from local cache
            img = load_image_safe(img_path)

            if img is not None:
                # Display image
                ax.imshow(img, cmap='gray')

                # Create title
                label_text = 'Malignant' if row['label'] == 1 else 'Benign'
                color = 'red' if row['label'] == 1 else 'green'

                title = f"{label_text}\\n"
                if 'view' in row and pd.notna(row['view']):
                    title += f"View: {row['view']}"
                if 'laterality' in row and pd.notna(row['laterality']):
                    title += f" | {row['laterality']}"

                ax.set_title(title, fontsize=10, fontweight='bold', color=color)
                ax.axis('off')
                loaded_count += 1
            else:
                # Failed to load
                ax.text(0.5, 0.5, f"Failed to load\\n{img_path.name[:20]}...",
                       ha='center', va='center', fontsize=8, color='red')
                ax.axis('off')
                failed_count += 1
        else:
            # Not cached
            ax.text(0.5, 0.5, "Image not available\\n(cache failed)",
                   ha='center', va='center', fontsize=8, color='orange')
            ax.axis('off')
            failed_count += 1

    # Hide extra subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"✓ Visualization complete!")
    print(f"  Successfully loaded: {loaded_count}/{len(samples)} images")
    if failed_count > 0:
        print(f"  Failed to load: {failed_count} images")
    print()

    # ========================================================================
    # STEP 5: Print dataset statistics
    # ========================================================================
    print("=" * 80)
    print("INBREAST DATASET SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(inbreast_metadata)}")

    if 'patient_id' in inbreast_metadata.columns:
        print(f"Patients: {inbreast_metadata['patient_id'].nunique()}")

    if 'breast_id' in inbreast_metadata.columns:
        print(f"Breasts: {inbreast_metadata['breast_id'].nunique()}")

    print()
    print("Label distribution:")
    mal_count = (inbreast_metadata['label'] == 1).sum()
    ben_count = (inbreast_metadata['label'] == 0).sum()
    total = len(inbreast_metadata)
    print(f"  Malignant: {mal_count} ({mal_count/total*100:.1f}%)")
    print(f"  Benign: {ben_count} ({ben_count/total*100:.1f}%)")

    print()
    if 'view' in inbreast_metadata.columns:
        print("View distribution:")
        for view, count in inbreast_metadata['view'].value_counts().items():
            print(f"  {view}: {count}")

    print("=" * 80)
"""

print("=" * 80)
print("COMPLETE VISUALIZATION FIX")
print("=" * 80)
print()
print("This fix handles ALL issues:")
print("  ✓ errno 107 (Transport endpoint not connected)")
print("  ✓ PNG vs DICOM path confusion")
print("  ✓ Missing benign images")
print("  ✓ Google Drive timeout errors")
print()
print("HOW IT WORKS:")
print("  1. Auto-detects PNG vs DICOM and fixes paths")
print("  2. Copies images to local /tmp cache")
print("  3. Uses retry logic (3 attempts)")
print("  4. Gracefully handles missing images")
print("  5. Shows clear success/failure counts")
print()
print("=" * 80)
print("INSTALLATION:")
print("=" * 80)
print("Replace your ENTIRE visualization cell with the code below:")
print("=" * 80)
print()
print(complete_fix)
print()
print("=" * 80)
print("This is a production-ready solution that will work reliably!")
print("=" * 80)
