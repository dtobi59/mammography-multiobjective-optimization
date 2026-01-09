# Quick Fix: FileNotFoundError for .dcm in PNG folder

## Problem
```
FileNotFoundError: [Errno 2] No such file or directory:
'/content/drive/MyDrive/INbreast/images_png/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm'
```

After converting DICOM to PNG, the metadata still references `.dcm` files but they're now `.png`.

## Quickest Fix (30 seconds)

Add this **ONE-LINE FIX** right after loading your metadata:

```python
# After: inbreast_metadata = load_metadata(...)
# Add this ONE line:
inbreast_metadata['image_path'] = inbreast_metadata['image_path'].str.replace('.dcm', '.png', case=False)
```

That's it! This changes all `.dcm` extensions to `.png` in the metadata.

## Complete Fix Code (if above doesn't work)

Add this cell BEFORE visualization:

```python
# Fix image path extensions from .dcm to .png
print("Fixing image paths...")

import pandas as pd
from pathlib import Path

# Update extensions
if 'image_path' in inbreast_metadata.columns:
    for idx, row in inbreast_metadata.iterrows():
        if pd.notna(row['image_path']):
            path_str = str(row['image_path'])
            # Replace .dcm with .png (case insensitive)
            if path_str.lower().endswith('.dcm'):
                inbreast_metadata.at[idx, 'image_path'] = path_str[:-4] + '.png'

    print(f"✓ Updated paths to .png format")

    # Verify a few files exist
    image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]
    sample_path = image_dir / inbreast_metadata.iloc[0]['image_path']
    if sample_path.exists():
        print(f"✓ Verified: {sample_path.name}")
    else:
        print(f"⚠ Not found: {sample_path.name}")
```

## Why This Happens

1. Original files: `20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm`
2. After conversion: `20586908_6c613a14b80a8591_MG_R_CC_ANON.png`
3. But metadata still says: `.dcm`
4. Result: FileNotFoundError

## Prevention

After running DICOM conversion, always update the config AND metadata:

```python
# After DICOM->PNG conversion, run:
config.INBREAST_CONFIG['image_dir'] = 'images_png'  # ✓
inbreast_metadata['image_path'] = inbreast_metadata['image_path'].str.replace('.dcm', '.png')  # ✓
```
