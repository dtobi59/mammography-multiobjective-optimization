"""
Fix for FileNotFoundError when code looks for .dcm but files are .png

This happens after DICOM to PNG conversion - the image_path column still
references .dcm files but they've been converted to .png
"""

fix_code = """
# ============================================================================
# FIX: Auto-detect PNG vs DICOM files after conversion
# Add this cell BEFORE visualization
# ============================================================================

from pathlib import Path
import pandas as pd

print("=" * 80)
print("FIXING IMAGE PATH EXTENSIONS (DCM -> PNG)")
print("=" * 80)
print()

# Check what format the files actually are
image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]
print(f"Image directory: {image_dir}")
print()

# Count files by extension
png_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.PNG"))
dcm_files = list(image_dir.glob("*.dcm")) + list(image_dir.glob("*.DCM"))

print(f"Found {len(png_files)} PNG files")
print(f"Found {len(dcm_files)} DICOM files")
print()

# Determine which format to use
if len(png_files) > 0 and len(dcm_files) == 0:
    print("✓ Using PNG format (converted)")
    needs_fix = True
elif len(dcm_files) > 0 and len(png_files) == 0:
    print("✓ Using DICOM format (original)")
    needs_fix = False
else:
    print("⚠ Mixed formats detected - using PNG if available")
    needs_fix = True

# Fix image paths if needed
if needs_fix:
    print()
    print("Updating image_path extensions from .dcm to .png...")

    fixed_count = 0
    for idx, row in inbreast_metadata.iterrows():
        if 'image_path' in row and pd.notna(row['image_path']):
            img_path = str(row['image_path'])

            # Change .dcm to .png (case insensitive)
            if img_path.lower().endswith('.dcm'):
                new_path = img_path[:-4] + '.png'
                inbreast_metadata.at[idx, 'image_path'] = new_path
                fixed_count += 1

    print(f"✓ Updated {fixed_count} paths from .dcm to .png")

    # Verify files exist
    missing_count = 0
    for idx, row in inbreast_metadata.iterrows():
        if 'image_path' in row and pd.notna(row['image_path']):
            full_path = image_dir / row['image_path']
            if not full_path.exists():
                missing_count += 1
                if missing_count <= 3:  # Show first 3 missing
                    print(f"  ⚠ Missing: {row['image_path']}")

    if missing_count > 0:
        print(f"  ⚠ WARNING: {missing_count} files not found")
    else:
        print("  ✓ All files verified")
else:
    print("No path updates needed")

print()
print("=" * 80)
print("SAMPLE IMAGE PATHS (first 5):")
print("=" * 80)
for idx, row in inbreast_metadata.head(5).iterrows():
    img_path = row.get('image_path', 'N/A')
    if img_path != 'N/A':
        full_path = image_dir / img_path
        exists = "✓" if full_path.exists() else "✗"
        print(f"  {exists} {img_path}")
print("=" * 80)
"""

print("=" * 80)
print("FIX FOR: FileNotFoundError looking for .dcm in PNG folder")
print("=" * 80)
print()
print("PROBLEM:")
print("  After DICOM->PNG conversion, files are .png but metadata still")
print("  references .dcm extensions, causing FileNotFoundError")
print()
print("SOLUTION:")
print("  Add this cell BEFORE visualization to auto-detect and fix paths")
print()
print("=" * 80)
print("CODE TO ADD AS NEW CELL:")
print("=" * 80)
print()
print(fix_code)
print()
print("=" * 80)
print()
print("This will:")
print("  1. Detect which format (PNG or DICOM) is in the directory")
print("  2. Automatically update image_path from .dcm to .png if needed")
print("  3. Verify all files exist")
print("  4. Show sample paths for verification")
print()
print("After running this, your visualization cell will work correctly!")
print("=" * 80)
