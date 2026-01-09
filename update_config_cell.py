"""
Update the config cell to also fix metadata paths after DICOM conversion.
"""

import json
from pathlib import Path

# Read notebook
notebook_path = Path("zero_shot_evaluation.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# New improved config update cell
improved_config_cell = """# Update config to use PNG images AND fix metadata paths
from pathlib import Path
import pandas as pd

print("=" * 80)
print("POST-CONVERSION SETUP")
print("=" * 80)
print()

# Step 1: Update config.py
print("Step 1: Updating config.py...")
with open('config.py', 'r') as f:
    config_content = f.read()

# Update image directory to point to converted PNGs
updated = False
if '"image_dir": "images"' in config_content:
    config_content = config_content.replace(
        '"image_dir": "images"',
        '"image_dir": "images_png"'
    )
    updated = True
elif '"image_dir": "AllDICOMs"' in config_content:
    config_content = config_content.replace(
        '"image_dir": "AllDICOMs"',
        '"image_dir": "images_png"'
    )
    updated = True

if updated:
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("  ✓ Updated config.py to use images_png directory")
else:
    print("  ✓ Config already set to images_png")

# Step 2: Reload config
import importlib
importlib.reload(config)
print(f"  Current image_dir: {config.INBREAST_CONFIG['image_dir']}")
print()

# Step 3: Fix metadata paths (.dcm -> .png)
print("Step 2: Fixing metadata paths...")
if 'image_path' in inbreast_metadata.columns:
    # Count current extensions
    dcm_count = inbreast_metadata['image_path'].astype(str).str.lower().str.endswith('.dcm').sum()
    png_count = inbreast_metadata['image_path'].astype(str).str.lower().str.endswith('.png').sum()

    print(f"  Current paths: {dcm_count} .dcm, {png_count} .png")

    if dcm_count > 0:
        # Update all .dcm to .png
        inbreast_metadata['image_path'] = inbreast_metadata['image_path'].astype(str).str.replace(
            '.dcm', '.png', case=False, regex=False
        ).str.replace(
            '.DCM', '.PNG', case=False, regex=False
        )

        # Verify
        dcm_after = inbreast_metadata['image_path'].astype(str).str.lower().str.endswith('.dcm').sum()
        png_after = inbreast_metadata['image_path'].astype(str).str.lower().str.endswith('.png').sum()

        print(f"  ✓ Updated to: {dcm_after} .dcm, {png_after} .png")
    else:
        print("  ✓ Paths already use .png extension")
else:
    print("  ⚠ No image_path column in metadata")
print()

# Step 4: Verify files exist
print("Step 3: Verifying files...")
image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]

if image_dir.exists():
    png_files = list(image_dir.glob("*.png"))
    print(f"  Found {len(png_files)} PNG files in {image_dir.name}/")

    # Check a few files from metadata
    if len(inbreast_metadata) > 0 and 'image_path' in inbreast_metadata.columns:
        sample_size = min(5, len(inbreast_metadata))
        found_count = 0
        missing_count = 0

        for img_path in inbreast_metadata['image_path'].head(sample_size):
            full_path = image_dir / str(img_path)
            if full_path.exists():
                found_count += 1
            else:
                missing_count += 1
                if missing_count <= 2:  # Show first 2 missing
                    print(f"    ⚠ Not found: {img_path}")

        if missing_count == 0:
            print(f"  ✓ All checked files exist ({found_count}/{sample_size})")
        else:
            print(f"  ⚠ {found_count}/{sample_size} files found, {missing_count} missing")
else:
    print(f"  ⚠ Directory not found: {image_dir}")

print()
print("=" * 80)
print("SETUP COMPLETE")
print("=" * 80)
print(f"Config: {config.INBREAST_CONFIG['image_dir']}")
print(f"Metadata: {len(inbreast_metadata)} images")
if 'image_path' in inbreast_metadata.columns:
    sample_path = str(inbreast_metadata['image_path'].iloc[0])
    print(f"Sample path: {sample_path}")
print()
print("✓ Ready for visualization and evaluation!")
print("=" * 80)
"""

# Find and update the cell
for cell in notebook['cells']:
    if cell.get('id') == 'pgt4cwarftr':
        print(f"Updating cell: pgt4cwarftr (Post-conversion config)")
        cell['source'] = improved_config_cell
        break

# Write updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print()
print("="* 80)
print("Notebook updated successfully!")
print("=" * 80)
print()
print("Updated cell 'pgt4cwarftr' to:")
print("  1. Update config.py to use images_png")
print("  2. Fix metadata paths (.dcm -> .png)")
print("  3. Verify files exist")
print("  4. Show comprehensive status")
print("=" * 80)
