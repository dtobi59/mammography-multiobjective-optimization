"""
Smart image loader that auto-detects PNG vs DICOM regardless of extension in metadata.
Replaces the load_image function to handle both formats intelligently.
"""

improved_loader = """
def load_image_smart(img_path, max_retries=3):
    \"\"\"
    Smart image loader that handles both PNG and DICOM.
    Auto-detects format by trying both extensions if original fails.
    \"\"\"
    from pathlib import Path
    import numpy as np
    from PIL import Image
    import time

    def try_load(path_to_try):
        \"\"\"Try loading a specific path.\"\"\"
        if not path_to_try.exists():
            return None

        # Detect format by actual extension
        ext = path_to_try.suffix.lower()

        if ext in ['.dcm', '.dicom']:
            # Load DICOM
            import pydicom
            dcm = pydicom.dcmread(str(path_to_try))
            img_array = dcm.pixel_array.astype(float)

            # Apply PhotometricInterpretation if needed
            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    img_array = img_array.max() - img_array

            # Normalize to 0-255
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = (img_array - img_min) / (img_max - img_min) * 255.0
            else:
                img_array = np.zeros_like(img_array)

            img_array = img_array.astype(np.uint8)
            return Image.fromarray(img_array, mode='L')
        else:
            # Load standard image (PNG, JPEG, etc.)
            return Image.open(path_to_try).convert('L')

    # Try loading with retries
    for attempt in range(max_retries):
        try:
            # First, try the path as given
            result = try_load(img_path)
            if result:
                return result

            # If that fails, try alternative extensions
            img_path_obj = Path(img_path)
            stem = img_path_obj.stem
            parent = img_path_obj.parent

            # Try common extensions
            for ext in ['.png', '.PNG', '.dcm', '.DCM', '.jpg', '.jpeg']:
                alt_path = parent / f"{stem}{ext}"
                result = try_load(alt_path)
                if result:
                    return result

            # If all attempts failed, raise error
            raise FileNotFoundError(f"Could not find image with stem '{stem}' in {parent}")

        except (OSError, IOError) as e:
            error_str = str(e)
            if "107" in error_str or "Transport endpoint" in error_str:
                # Drive connection issue - retry
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            raise e

    raise Exception(f"Failed to load {img_path} after {max_retries} retries")
"""

visualization_with_smart_loader = """
# ============================================================================
# IMPROVED VISUALIZATION WITH SMART IMAGE LOADER
# Replace your entire visualization cell with this
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import time

# Try to import pydicom for DICOM support
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    print("⚠ pydicom not installed. Installing...")
    !pip install -q pydicom
    import pydicom
    PYDICOM_AVAILABLE = True

def load_image_smart(img_path, max_retries=3):
    \"\"\"
    Smart image loader - auto-detects PNG vs DICOM.
    Tries alternative extensions if the specified one doesn't exist.
    \"\"\"
    def try_load(path_to_try):
        if not path_to_try.exists():
            return None

        ext = path_to_try.suffix.lower()
        if ext in ['.dcm', '.dicom']:
            # Load DICOM
            dcm = pydicom.dcmread(str(path_to_try))
            img_array = dcm.pixel_array.astype(float)

            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    img_array = img_array.max() - img_array

            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = (img_array - img_min) / (img_max - img_min) * 255.0
            else:
                img_array = np.zeros_like(img_array)

            return Image.fromarray(img_array.astype(np.uint8), mode='L')
        else:
            return Image.open(path_to_try).convert('L')

    # Try with retries
    for attempt in range(max_retries):
        try:
            # Try as-is
            result = try_load(img_path)
            if result:
                return result

            # Try alternative extensions
            stem = img_path.stem
            parent = img_path.parent
            for ext in ['.png', '.PNG', '.dcm', '.DCM']:
                alt_path = parent / f"{stem}{ext}"
                result = try_load(alt_path)
                if result:
                    return result

            raise FileNotFoundError(f"No image found for: {stem}")

        except (OSError, IOError) as e:
            if "107" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

print("=" * 80)
print("INBREAST DATASET VISUALIZATION")
print("=" * 80)
print()

if len(inbreast_metadata) == 0:
    print("⚠ ERROR: No images in metadata!")
else:
    # Select samples
    malignant_samples = inbreast_metadata[inbreast_metadata['label'] == 1].head(4)
    benign_samples = inbreast_metadata[inbreast_metadata['label'] == 0].head(4)
    samples = pd.concat([malignant_samples, benign_samples])

    if len(samples) == 0:
        print("⚠ WARNING: No samples available")
    else:
        print(f"Showing {len(samples)} sample images:")
        print(f"  Malignant: {len(malignant_samples)}")
        print(f"  Benign: {len(benign_samples)}")
        print()

        # Create figure
        n_images = len(samples)
        n_cols = min(4, n_images)
        n_rows = max(1, (n_images + n_cols - 1) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        fig.suptitle('Sample INbreast Mammography Images', fontsize=16, fontweight='bold')

        image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]

        for idx, (ax, (_, row)) in enumerate(zip(axes, samples.iterrows())):
            if 'image_path' in row and pd.notna(row['image_path']):
                try:
                    img_path = image_dir / str(row['image_path'])

                    # Use smart loader - handles PNG/DICOM auto-detection
                    img = load_image_smart(img_path)

                    # Display
                    ax.imshow(img, cmap='gray')

                    # Title
                    label_text = 'Malignant' if row.get('label') == 1 else 'Benign'
                    color = 'red' if row.get('label') == 1 else 'green'

                    title = f"{label_text}\\n"
                    if 'view' in row and pd.notna(row['view']):
                        title += f"View: {row['view']}"
                    if 'patient_id' in row and pd.notna(row['patient_id']):
                        title += f"\\nID: {str(row['patient_id'])[:8]}"

                    ax.set_title(title, fontsize=10, fontweight='bold', color=color)
                    ax.axis('off')

                except Exception as e:
                    error_msg = str(e)[:50]
                    ax.text(0.5, 0.5, f'Error:\\n{error_msg}',
                            ha='center', va='center', fontsize=8, wrap=True)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No image path',
                        ha='center', va='center', fontsize=8)
                ax.axis('off')

        # Hide extra subplots
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

        # Print statistics
        print()
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
        mal = (inbreast_metadata['label'] == 1).sum()
        ben = (inbreast_metadata['label'] == 0).sum()
        print(f"  Malignant: {mal} ({mal / len(inbreast_metadata) * 100:.1f}%)")
        print(f"  Benign: {ben} ({ben / len(inbreast_metadata) * 100:.1f}%)")
        print()
        if 'view' in inbreast_metadata.columns:
            print("View distribution:")
            print(inbreast_metadata['view'].value_counts())
        print("=" * 80)
"""

print("=" * 80)
print("SMART IMAGE LOADER - AUTO-DETECTS PNG vs DICOM")
print("=" * 80)
print()
print("PROBLEM:")
print("  FileNotFoundError: looking for .dcm in folder with .png files")
print()
print("ROOT CAUSE:")
print("  After DICOM->PNG conversion, metadata still references .dcm")
print("  but files are now .png")
print()
print("SOLUTION:")
print("  Smart image loader that:")
print("  1. Tries the path as specified")
print("  2. If not found, tries alternative extensions (.png, .dcm)")
print("  3. Loads whichever file actually exists")
print()
print("=" * 80)
print("OPTION 1: Just the load_image function")
print("=" * 80)
print()
print(improved_loader)
print()
print("=" * 80)
print("OPTION 2: Complete visualization cell (RECOMMENDED)")
print("=" * 80)
print()
print(visualization_with_smart_loader)
print()
print("=" * 80)
print("USAGE:")
print("  Replace your entire visualization cell with 'OPTION 2' code above")
print("  This will handle both PNG and DICOM automatically!")
print("=" * 80)
