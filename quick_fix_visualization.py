"""
Quick fix script for errno 107 visualization errors.
This can be added as a new cell BEFORE the visualization cell in the notebook.
"""

quick_fix_code = '''
# ============================================================================
# QUICK FIX: Add this cell BEFORE your visualization cell
# ============================================================================
# This fixes errno 107 "Transport endpoint is not connected" errors

import shutil
import time
from pathlib import Path
from google.colab import drive

def fix_drive_connection():
    """Check and fix Google Drive connection."""
    try:
        drive_test = Path("/content/drive/MyDrive")
        list(drive_test.glob("*"))  # Test connection
        print("✓ Google Drive connection OK")
        return True
    except Exception as e:
        if "107" in str(e) or "Transport endpoint" in str(e):
            print("⚠ Reconnecting Google Drive...")
            try:
                drive.mount('/content/drive', force_remount=True)
                print("✓ Google Drive reconnected")
                return True
            except:
                print("✗ Failed to reconnect Google Drive")
                return False
        print(f"✗ Drive error: {str(e)[:50]}")
        return False

def cache_images_locally(metadata, image_dir, max_images=8):
    """Copy images to local /tmp to avoid repeated Drive access."""
    cache_dir = Path("/tmp/viz_cache")
    cache_dir.mkdir(exist_ok=True)

    print(f"Caching up to {max_images} images locally...")

    # Select samples
    malignant = metadata[metadata['label'] == 1].head(max_images // 2)
    benign = metadata[metadata['label'] == 0].head(max_images // 2)
    samples = pd.concat([malignant, benign])

    cached_paths = {}
    success_count = 0

    for idx, row in samples.iterrows():
        if 'image_path' not in row or pd.isna(row['image_path']):
            continue

        source_path = Path(image_dir) / str(row['image_path'])
        cache_path = cache_dir / source_path.name

        # Try copying with retries
        for attempt in range(3):
            try:
                if not cache_path.exists():
                    shutil.copy2(str(source_path), str(cache_path))
                cached_paths[row['image_id']] = cache_path
                success_count += 1
                break
            except (OSError, IOError) as e:
                if "107" in str(e):
                    print(f"  Retry {attempt + 1}/3: {source_path.name}")
                    fix_drive_connection()
                    time.sleep(1)
                else:
                    print(f"  Skip {source_path.name}: {str(e)[:30]}")
                    break

    print(f"✓ Cached {success_count}/{len(samples)} images to {cache_dir}")
    return samples, cached_paths, cache_dir

# Run the fix
print("=" * 60)
print("FIXING GOOGLE DRIVE CONNECTION")
print("=" * 60)
fix_drive_connection()

# Cache images
inbreast_image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]
viz_samples, viz_cached_paths, viz_cache_dir = cache_images_locally(
    inbreast_metadata,
    inbreast_image_dir,
    max_images=8
)

print("=" * 60)
print("✓ Ready for visualization!")
print("=" * 60)
'''

print("=" * 80)
print("QUICK FIX FOR ERRNO 107 VISUALIZATION ERROR")
print("=" * 80)
print()
print("PROBLEM:")
print("  When visualizing images from Google Drive, you may see:")
print("  'OSError: [Errno 107] Transport endpoint is not connected'")
print()
print("SOLUTION:")
print("  Add the code below as a NEW CELL before your visualization cell.")
print("  It will:")
print("  1. Check Google Drive connection and reconnect if needed")
print("  2. Copy images to local /tmp (faster and more reliable)")
print("  3. Retry failed operations automatically")
print()
print("=" * 80)
print("STEP 1: Add this cell BEFORE your visualization cell")
print("=" * 80)
print()
print(quick_fix_code)
print()
print("=" * 80)
print("STEP 2: Modify your visualization cell")
print("=" * 80)
print()
print("Replace this section in your visualization cell:")
print()
print("  # OLD:")
print("  samples = pd.concat([malignant_samples, benign_samples])")
print("  inbreast_image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG['image_dir']")
print()
print("  # NEW:")
print("  samples = viz_samples  # Use pre-cached samples")
print("  inbreast_image_dir = viz_cache_dir  # Use local cache directory")
print()
print("=" * 80)
print("ALTERNATIVE: Simpler fix (if above doesn't work)")
print("=" * 80)
print()
print("Just reduce the number of images to visualize:")
print()
print("  # Change from 4+4 to 2+2")
print("  malignant_samples = inbreast_metadata[inbreast_metadata['label'] == 1].head(2)")
print("  benign_samples = inbreast_metadata[inbreast_metadata['label'] == 0].head(2)")
print()
print("This reduces Drive access and makes errors less likely.")
print("=" * 80)
