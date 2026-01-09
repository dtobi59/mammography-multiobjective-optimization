"""
Fix for errno 107 "Transport endpoint is not connected" errors in visualization.

This script provides an improved visualization cell with:
1. Google Drive connection check
2. Retry logic with exponential backoff
3. Local caching of images to avoid repeated Drive access
4. Better error handling for filesystem issues
"""

improved_visualization_code = """
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import time
import shutil
from google.colab import drive

# Try to import pydicom for DICOM support
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("⚠ pydicom not installed. Installing...")
    !pip install -q pydicom
    import pydicom
    PYDICOM_AVAILABLE = True

print("=" * 80)
print("INBREAST DATASET VISUALIZATION (WITH RETRY LOGIC)")
print("=" * 80)
print()

# Check Google Drive connection
def check_drive_connection():
    \"\"\"Check if Google Drive is still connected.\"\"\"
    try:
        drive_path = Path("/content/drive/MyDrive")
        if not drive_path.exists():
            print("⚠ Google Drive not mounted. Attempting to remount...")
            drive.mount('/content/drive', force_remount=True)
            return True
        # Try to list directory to verify connection
        list(drive_path.glob("*"))
        return True
    except (OSError, Exception) as e:
        error_str = str(e)
        if "107" in error_str or "Transport endpoint" in error_str:
            print("⚠ Google Drive connection lost. Attempting to remount...")
            try:
                drive.mount('/content/drive', force_remount=True)
                return True
            except:
                return False
        return True

def load_image_with_retry(img_path, max_retries=3):
    \"\"\"Load image from either PNG or DICOM format with retry logic.\"\"\"
    for attempt in range(max_retries):
        try:
            if img_path.suffix.lower() in ['.dcm', '.dicom']:
                # Load DICOM file
                dcm = pydicom.dcmread(str(img_path))
                img_array = dcm.pixel_array

                # Normalize to 0-255 range
                img_array = img_array.astype(float)
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
                img_array = img_array.astype(np.uint8)

                return Image.fromarray(img_array, mode='L')
            else:
                # Load standard image format (PNG, JPEG, etc.)
                return Image.open(img_path).convert('L')
        except (OSError, IOError) as e:
            error_str = str(e)
            if "107" in error_str or "Transport endpoint" in error_str:
                print(f"  Retry {attempt + 1}/{max_retries} for {img_path.name} (Drive disconnected)")
                # Check and restore Drive connection
                if not check_drive_connection():
                    raise Exception("Could not restore Google Drive connection")
                # Wait before retry (exponential backoff)
                time.sleep(2 ** attempt)
            else:
                raise e
    raise Exception(f"Failed to load {img_path} after {max_retries} retries")

def copy_images_to_local(samples, image_dir):
    \"\"\"Copy sample images to local storage to avoid repeated Drive access.\"\"\"
    local_cache_dir = Path("/tmp/inbreast_viz_cache")
    local_cache_dir.mkdir(exist_ok=True)

    print("Copying sample images to local storage...")
    local_paths = []
    failed = []

    for idx, row in samples.iterrows():
        if 'image_path' in row and pd.notna(row['image_path']):
            try:
                img_path = image_dir / str(row['image_path'])
                local_path = local_cache_dir / img_path.name

                # Copy with retry logic
                for attempt in range(3):
                    try:
                        if not local_path.exists():
                            shutil.copy2(str(img_path), str(local_path))
                        local_paths.append(local_path)
                        break
                    except (OSError, IOError) as e:
                        if "107" in str(e) or "Transport endpoint" in str(e):
                            print(f"  Retry {attempt + 1}/3 copying {img_path.name}")
                            check_drive_connection()
                            time.sleep(2 ** attempt)
                        else:
                            raise e
                else:
                    local_paths.append(None)
                    failed.append(img_path.name)
            except Exception as e:
                print(f"  Failed to copy {img_path.name}: {str(e)[:50]}")
                local_paths.append(None)
                failed.append(img_path.name)
        else:
            local_paths.append(None)

    if failed:
        print(f"⚠ Failed to copy {len(failed)} images")
    else:
        print("✓ All images copied successfully")
    print()

    return local_paths

# Check if we have any data
if len(inbreast_metadata) == 0:
    print("⚠ ERROR: No images in metadata!")
    print("The previous steps may have failed to load or match images.")
    print("Please check the output from the previous cells.")
    print("=" * 80)
else:
    # Verify Drive connection before starting
    print("Checking Google Drive connection...")
    if check_drive_connection():
        print("✓ Google Drive connected")
    else:
        print("⚠ WARNING: Google Drive connection issues detected")
    print()

    # Select sample images (4 malignant, 4 benign if available)
    malignant_samples = inbreast_metadata[inbreast_metadata['label'] == 1].head(4)
    benign_samples = inbreast_metadata[inbreast_metadata['label'] == 0].head(4)

    # Combine for visualization
    samples = pd.concat([malignant_samples, benign_samples])

    if len(samples) == 0:
        print("⚠ WARNING: No samples available for visualization")
        print(f"Total images in metadata: {len(inbreast_metadata)}")
        print(f"Malignant (label=1): {(inbreast_metadata['label'] == 1).sum()}")
        print(f"Benign (label=0): {(inbreast_metadata['label'] == 0).sum()}")
        print("\\nLabel distribution:")
        print(inbreast_metadata['label'].value_counts())
        print("=" * 80)
    else:
        print(f"Showing {len(samples)} sample images:")
        print(f"  Malignant: {len(malignant_samples)}")
        print(f"  Benign: {len(benign_samples)}")
        print()

        inbreast_image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]

        # Copy images to local storage first (avoids repeated Drive access)
        local_image_paths = copy_images_to_local(samples, inbreast_image_dir)

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

        for idx, (ax, (_, row), local_path) in enumerate(zip(axes, samples.iterrows(), local_image_paths)):
            if local_path and local_path.exists():
                try:
                    # Load from local cache (much faster and more reliable)
                    img = load_image_with_retry(local_path, max_retries=3)

                    # Display image
                    ax.imshow(img, cmap='gray')

                    # Create title with metadata
                    label_text = 'Malignant' if row.get('label') == 1 else 'Benign'
                    color = 'red' if row.get('label') == 1 else 'green'

                    title = f"{label_text}\\n"
                    if 'view' in row and pd.notna(row['view']):
                        title += f"View: {row['view']}"
                    if 'patient_id' in row and pd.notna(row['patient_id']):
                        title += f"\\nID: {row['patient_id']}"

                    ax.set_title(title, fontsize=10, fontweight='bold', color=color)
                    ax.axis('off')

                except Exception as e:
                    error_msg = str(e)
                    if len(error_msg) > 50:
                        error_msg = error_msg[:50] + "..."
                    ax.text(0.5, 0.5, f'Error loading:\\n{local_path.name}\\n{error_msg}',
                            ha='center', va='center', fontsize=8, wrap=True)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'Image not available:\\n(copy failed)',
                        ha='center', va='center', fontsize=8, color='red')
                ax.axis('off')

        # Hide extra subplots if any
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

        # Print dataset statistics
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
        print(f"  Malignant: {(inbreast_metadata['label'] == 1).sum()} ({(inbreast_metadata['label'] == 1).sum() / len(inbreast_metadata) * 100:.1f}%)")
        print(f"  Benign: {(inbreast_metadata['label'] == 0).sum()} ({(inbreast_metadata['label'] == 0).sum() / len(inbreast_metadata) * 100:.1f}%)")
        print()
        if 'view' in inbreast_metadata.columns:
            print("View distribution:")
            print(inbreast_metadata['view'].value_counts())
        print("=" * 80)
"""

print("=" * 80)
print("IMPROVED VISUALIZATION CODE FOR ERRNO 107 FIX")
print("=" * 80)
print()
print("This code includes:")
print("  1. Google Drive connection checking and auto-remount")
print("  2. Retry logic with exponential backoff (3 attempts)")
print("  3. Local caching of images (copies to /tmp first)")
print("  4. Better error handling for OSError errno 107")
print()
print("=" * 80)
print("CODE TO REPLACE IN NOTEBOOK:")
print("=" * 80)
print()
print(improved_visualization_code)
print()
print("=" * 80)
print()
print("USAGE INSTRUCTIONS:")
print("=" * 80)
print("1. Open zero_shot_evaluation.ipynb in Colab")
print("2. Find the visualization cell (Section 4a)")
print("3. Replace the cell content with the code above")
print("4. Run the cell - it will now handle Drive disconnections")
print()
print("BENEFITS:")
print("  - Automatically detects and recovers from Drive disconnections")
print("  - Copies images to local storage first (faster, more reliable)")
print("  - Retries failed operations up to 3 times")
print("  - Shows clear error messages if issues persist")
print("=" * 80)
