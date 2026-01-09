"""
Diagnostic script to understand why benign images are failing with errno 107.
Run this BEFORE the visualization to identify issues.
"""

diagnostic_code = """
# ============================================================================
# DIAGNOSTIC: Run this cell BEFORE visualization to identify issues
# ============================================================================

import pandas as pd
from pathlib import Path
import os

print("=" * 80)
print("DIAGNOSTIC: INVESTIGATING BENIGN IMAGE ERRORS")
print("=" * 80)
print()

# Check metadata
print("1. Metadata Check:")
print(f"   Total images in metadata: {len(inbreast_metadata)}")
print(f"   Malignant (label=1): {(inbreast_metadata['label'] == 1).sum()}")
print(f"   Benign (label=0): {(inbreast_metadata['label'] == 0).sum()}")
print()

# Check image paths
print("2. Image Path Check:")
image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]
print(f"   Image directory: {image_dir}")
print(f"   Directory exists: {image_dir.exists()}")
print()

# Check benign image paths specifically
print("3. Benign Image Paths (first 10):")
benign_samples = inbreast_metadata[inbreast_metadata['label'] == 0].head(10)

for idx, row in benign_samples.iterrows():
    if 'image_path' in row and pd.notna(row['image_path']):
        img_path = image_dir / str(row['image_path'])

        # Check existence
        try:
            exists = img_path.exists()
            status = "✓" if exists else "✗"

            # Check alternative extensions
            if not exists:
                alt_png = img_path.with_suffix('.png')
                alt_dcm = img_path.with_suffix('.dcm')
                if alt_png.exists():
                    status = f"⚠ (exists as .png)"
                elif alt_dcm.exists():
                    status = f"⚠ (exists as .dcm)"

            print(f"   {status} {img_path.name}")

        except (OSError, Exception) as e:
            error_str = str(e)
            if "107" in error_str:
                print(f"   ✗ {img_path.name} - ERRNO 107 (Drive disconnected)")
            else:
                print(f"   ✗ {img_path.name} - {str(e)[:30]}")
    else:
        print(f"   ✗ Row {idx}: No image_path")

print()

# Check file extensions in directory
print("4. File Extension Analysis:")
all_files = list(image_dir.glob("*.*"))[:20]  # First 20 files
extensions = {}
for f in all_files:
    ext = f.suffix.lower()
    extensions[ext] = extensions.get(ext, 0) + 1

for ext, count in sorted(extensions.items()):
    print(f"   {ext}: {count} files")
print()

# Check metadata paths vs actual files
print("5. Path Format Check:")
if len(inbreast_metadata) > 0:
    sample_path = str(inbreast_metadata.iloc[0].get('image_path', 'N/A'))
    print(f"   Metadata path format: {sample_path}")
    print(f"   Extension in metadata: {Path(sample_path).suffix}")

    if len(all_files) > 0:
        actual_file = all_files[0].name
        print(f"   Actual file format: {actual_file}")
        print(f"   Extension in directory: {all_files[0].suffix}")

        if Path(sample_path).suffix != all_files[0].suffix:
            print(f"   ⚠ MISMATCH: Metadata has {Path(sample_path).suffix} but files are {all_files[0].suffix}")
        else:
            print(f"   ✓ Extensions match")
print()

# Google Drive connection test
print("6. Google Drive Connection Test:")
try:
    # Try to list files (this will fail with errno 107 if Drive is disconnected)
    test_files = list(image_dir.glob("*.*"))[:5]
    print(f"   ✓ Successfully listed {len(test_files)} files")
    print(f"   ✓ Google Drive connection OK")
except OSError as e:
    if "107" in str(e):
        print(f"   ✗ ERRNO 107: Google Drive disconnected!")
        print(f"   → Need to remount: drive.mount('/content/drive', force_remount=True)")
    else:
        print(f"   ✗ Error: {str(e)[:50]}")
print()

# Recommendations
print("=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)

recommendations = []

# Check for extension mismatch
if len(inbreast_metadata) > 0 and len(all_files) > 0:
    meta_ext = Path(str(inbreast_metadata.iloc[0].get('image_path', ''))).suffix
    actual_ext = all_files[0].suffix
    if meta_ext != actual_ext:
        recommendations.append(
            f"Fix extension mismatch: metadata has {meta_ext}, files are {actual_ext}\\n"
            f"   → Add: inbreast_metadata['image_path'] = inbreast_metadata['image_path'].str.replace('{meta_ext}', '{actual_ext}')"
        )

# Check for Drive errors
try:
    list(image_dir.glob("*.*"))[:1]
except OSError as e:
    if "107" in str(e):
        recommendations.append(
            "Remount Google Drive:\\n"
            "   → from google.colab import drive\\n"
            "   → drive.mount('/content/drive', force_remount=True)"
        )

# General recommendations
if not recommendations:
    recommendations.append("Use local caching to avoid repeated Drive access")
    recommendations.append("Use the complete_visualization_fix.py code")

for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")
    print()

print("=" * 80)
"""

print("=" * 80)
print("DIAGNOSTIC SCRIPT FOR VISUALIZATION ERRORS")
print("=" * 80)
print()
print("PURPOSE:")
print("  Identifies WHY benign images are failing with errno 107")
print()
print("WHAT IT CHECKS:")
print("  1. Metadata completeness")
print("  2. File paths for benign images")
print("  3. Extension mismatches (.dcm vs .png)")
print("  4. Google Drive connection status")
print("  5. Actual files in directory")
print()
print("=" * 80)
print("HOW TO USE:")
print("=" * 80)
print("1. Add this as a NEW CELL before your visualization")
print("2. Run it to see detailed diagnostics")
print("3. Follow the recommendations it provides")
print()
print("=" * 80)
print("CODE:")
print("=" * 80)
print()
print(diagnostic_code)
print()
print("=" * 80)
