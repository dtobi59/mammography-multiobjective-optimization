# Corrupted PNG Files - Detection and Repair

## Problem

You may encounter this error when loading images:

```python
PIL.UnidentifiedImageError: cannot identify image file '/path/to/image.png'
```

This happens when:
- PNG files exist but are corrupted
- DICOM to PNG conversion was interrupted
- Google Drive sync created placeholder files
- Files were partially written during conversion

## Solution

We've added automatic detection and repair for corrupted PNG files.

### Method 1: Re-run the Conversion Cell (Recommended)

The DICOM conversion cell now automatically:

1. **Validates existing PNGs** before skipping them
2. **Detects corrupted files** during the scan
3. **Removes corrupted files** automatically
4. **Reconverts them** from the original DICOM

Just re-run the conversion cell in the notebook:
- It will say: `[!] Removing corrupted file: filename.png`
- Then reconvert it automatically
- Validate the new file to ensure it's readable

### Method 2: Use the Repair Script

For manual repairs or batch processing:

```bash
python fix_corrupted_pngs.py
```

The script will:
1. Scan all PNG files in `images_png` directory
2. Attempt to open each one with PIL
3. Report which files are corrupted
4. Offer to reconvert them from DICOM

**Interactive Mode:**
```
CORRUPTED PNG FILE DETECTION AND REPAIR
================================================================================

Found 410 PNG files to check
Scanning for corrupted files...
100%|████████████████████████████████████████| 410/410 [00:12<00:00]

SCAN RESULTS
================================================================================
Total PNG files: 410
Valid files: 408
Corrupted files: 2

Corrupted files:
  1. 50997277_9054942f7be52dd9_MG_L_CC_ANON.png
  2. 50997298_abc123def456_MG_R_MLO_ANON.png

Would you like to reconvert these files from DICOM?
  1. Yes - Reconvert all corrupted files
  2. No - Just report and exit

Enter choice (1 or 2):
```

## How Validation Works

The validation function:

```python
def validate_png(png_path):
    """Check if PNG is valid and readable."""
    try:
        with Image.open(png_path) as img:
            img.load()  # Force load to detect corruption
        return True
    except:
        return False
```

This catches:
- File format errors
- Incomplete files
- Corrupted headers
- Invalid data

## Prevention

The conversion cell now includes **post-save validation**:

```python
# Save PNG
img.save(str(png_path), "PNG")

# VALIDATION: Verify the saved PNG is readable
if not validate_png(png_path):
    raise Exception("Saved PNG is corrupted or unreadable")
```

This ensures every PNG is valid before moving to the next file.

## Common Causes

### 1. Google Drive Sync Issues
**Symptom:** Files exist but are 0 bytes or show as "placeholder"

**Solution:**
- Re-run conversion cell (will reconvert)
- Check Google Drive storage space
- Ensure stable internet connection

### 2. Interrupted Conversion
**Symptom:** Some PNGs work, others don't

**Solution:**
- Re-run conversion cell (skips valid, reconverts corrupt)
- Check Colab runtime hasn't disconnected

### 3. Disk Space
**Symptom:** Later files are corrupted

**Solution:**
- Check Google Drive has space
- Clean up old files if needed

## Verification

After fixing corrupted files, verify:

```python
from pathlib import Path
from PIL import Image

png_dir = Path("/content/drive/MyDrive/INbreast/images_png")

# Quick check
for png in png_dir.glob("*.png"):
    try:
        with Image.open(png) as img:
            img.load()
    except Exception as e:
        print(f"Still corrupted: {png.name}")
```

## Prevention Best Practices

1. **Stable Connection:** Ensure good internet when converting
2. **Don't Interrupt:** Let conversion complete fully
3. **Check Storage:** Ensure Google Drive has enough space
4. **Use Validation:** Always use the updated conversion cell
5. **Monitor Output:** Watch for conversion errors

## Technical Details

### What Makes a PNG Corrupted?

- **Incomplete write:** File saved but data truncated
- **Invalid header:** PNG signature missing or wrong
- **Corrupt data:** IDAT chunks damaged
- **Zero bytes:** File created but no data written
- **Sync placeholder:** Google Drive creates stub file

### Why Validation is Important

Without validation:
- Corrupted files appear to exist
- Conversion is skipped (thinks it's done)
- Error only occurs during training/evaluation
- Hard to debug which file caused the issue

With validation:
- Corruption detected immediately
- File reconverted automatically
- No runtime errors during training
- Clear feedback on file status

## Summary

**The Fix:**
1. ✅ Automatic detection of corrupted PNGs
2. ✅ Automatic reconversion from DICOM
3. ✅ Post-save validation for quality
4. ✅ Clear reporting of file status
5. ✅ Interactive repair script

**What You Need to Do:**
1. Re-run the DICOM conversion cell
2. Check the output for "[!] Removing corrupted file" messages
3. Verify "All PNG files verified as readable" at the end
4. If issues persist, run `python fix_corrupted_pngs.py`

**Result:**
No more `UnidentifiedImageError` during training or evaluation!

---

**Note:** The original DICOM files are never modified. If a PNG is corrupted, we simply delete it and reconvert from the source DICOM. This is safe and reliable.
