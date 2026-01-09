"""
Cell to add RIGHT AFTER DICOM conversion to fix all metadata paths.
This ensures metadata is synchronized with the converted PNG files.
"""

fix_cell = """
# ============================================================================
# FIX: Update metadata paths after DICOM to PNG conversion
# Add this cell RIGHT AFTER the PNG conversion cell (before visualization)
# ============================================================================

from pathlib import Path
import pandas as pd

print("=" * 80)
print("UPDATING METADATA FOR PNG FILES")
print("=" * 80)
print()

# Update config to use images_png directory
with open('config.py', 'r') as f:
    config_content = f.read()

# Update to images_png
for old_dir in ['"image_dir": "images"', '"image_dir": "AllDICOMs"']:
    if old_dir in config_content:
        config_content = config_content.replace(old_dir, '"image_dir": "images_png"')

with open('config.py', 'w') as f:
    f.write(config_content)

# Reload config
import importlib
importlib.reload(config)

print(f"✓ Config updated: image_dir = {config.INBREAST_CONFIG['image_dir']}")
print()

# Fix all .dcm references in metadata to .png
if 'image_path' in inbreast_metadata.columns:
    # Count current extensions
    dcm_count = inbreast_metadata['image_path'].astype(str).str.lower().str.endswith('.dcm').sum()
    png_count = inbreast_metadata['image_path'].astype(str).str.lower().str.endswith('.png').sum()

    print(f"Current metadata paths:")
    print(f"  .dcm: {dcm_count}")
    print(f"  .png: {png_count}")
    print()

    if dcm_count > 0:
        print("Updating all .dcm paths to .png...")

        # Update paths - handle both .dcm and .DCM
        inbreast_metadata['image_path'] = inbreast_metadata['image_path'].astype(str).str.replace(
            '.dcm', '.png', case=False, regex=False
        ).str.replace(
            '.DCM', '.PNG', case=False, regex=False
        )

        # Verify update
        dcm_after = inbreast_metadata['image_path'].astype(str).str.lower().str.endswith('.dcm').sum()
        png_after = inbreast_metadata['image_path'].astype(str).str.lower().str.endswith('.png').sum()

        print(f"✓ Updated paths:")
        print(f"  .dcm: {dcm_after}")
        print(f"  .png: {png_after}")
        print()
    else:
        print("✓ Paths already use .png")
        print()

    # Verify files actually exist
    print("Verifying files exist...")
    image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]

    missing_count = 0
    existing_count = 0

    for img_path in inbreast_metadata['image_path'].head(10):  # Check first 10
        full_path = image_dir / str(img_path)
        if full_path.exists():
            existing_count += 1
        else:
            missing_count += 1
            if missing_count <= 3:
                print(f"  ⚠ Not found: {img_path}")

    if missing_count == 0:
        print(f"✓ All checked files exist ({existing_count}/{existing_count})")
    else:
        print(f"⚠ Some files missing: {existing_count} found, {missing_count} missing")
        print("  (Checked first 10 files only)")

print()
print("=" * 80)
print("METADATA UPDATE COMPLETE")
print("=" * 80)
print(f"Total images in metadata: {len(inbreast_metadata)}")
print(f"Image directory: {config.INBREAST_CONFIG['image_dir']}")
print()
print("✓ Ready for visualization!")
print("=" * 80)
"""

print("=" * 80)
print("FIX FOR: FileNotFoundError looking for .dcm in images_png")
print("=" * 80)
print()
print("PROBLEM:")
print("  After converting DICOM to PNG, metadata still references .dcm files")
print("  Error: '/content/drive/MyDrive/INbreast/images_png/...ANON.dcm'")
print()
print("SOLUTION:")
print("  Add this cell RIGHT AFTER the DICOM conversion cell")
print("  It will:")
print("  1. Update config.py to use 'images_png' directory")
print("  2. Change all .dcm extensions to .png in metadata")
print("  3. Verify files exist")
print()
print("=" * 80)
print("ADD THIS AS A NEW CELL AFTER DICOM CONVERSION:")
print("=" * 80)
print()
print(fix_cell)
print()
print("=" * 80)
print()
print("USAGE:")
print("  1. In your notebook, find the DICOM conversion cell")
print("  2. Add a NEW cell RIGHT AFTER it")
print("  3. Copy-paste the code above")
print("  4. Run the cell")
print("  5. Then run visualization - it will work!")
print("=" * 80)
