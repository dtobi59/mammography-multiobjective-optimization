# Visualization Error Fixes

## Problem Summary

When visualizing 8 sample images (4 malignant + 4 benign) in the zero_shot_evaluation notebook, some images (especially benign) show:
```
Error loading: errno 107 - Transport endpoint is not connected
```

Or:
```
FileNotFoundError: .../images_png/...ANON.dcm
```

## Root Causes

1. **errno 107**: Google Drive connection times out or disconnects during file reading
2. **FileNotFoundError**: Metadata references `.dcm` files but they've been converted to `.png`
3. **Sequential access**: Reading 8 files from Drive sequentially increases failure probability

## Solutions (Choose One)

### Solution 1: Quick One-Line Fix ⚡ (Try This First)

**Best for**: Quick fix, if you just want it to work

**Add this ONE line** after loading metadata:

```python
# After: inbreast_metadata = load_metadata(...)
inbreast_metadata['image_path'] = inbreast_metadata['image_path'].str.replace('.dcm', '.png', case=False)
```

**Time**: 30 seconds

---

### Solution 2: Diagnostic + Targeted Fix 🔍 (If Solution 1 Fails)

**Best for**: Understanding what's wrong before fixing

1. **Add diagnostic cell** (from `diagnose_visualization_errors.py`):
   - Checks your specific setup
   - Identifies exact issues
   - Provides targeted recommendations

2. **Follow the recommendations** it provides

**Time**: 2 minutes

---

### Solution 3: Complete Robust Fix 🛡️ (Recommended for Production)

**Best for**: Permanent, reliable solution

**Replace your entire visualization cell** with code from `complete_visualization_fix.py`

**Features**:
- ✅ Auto-detects PNG vs DICOM
- ✅ Fixes path extensions automatically
- ✅ Copies images to local cache (bypasses Drive issues)
- ✅ Retry logic (3 attempts per image)
- ✅ Graceful error handling
- ✅ Shows detailed success/failure counts

**Time**: 1 minute (copy-paste)

---

## File Guide

| File | Purpose | When to Use |
|------|---------|-------------|
| `QUICK_FIX_PNG_DCM.md` | One-line path fix | First attempt |
| `diagnose_visualization_errors.py` | Diagnostic tool | To understand the problem |
| `complete_visualization_fix.py` | Complete solution | For permanent fix |
| `fix_visualization_errors.py` | Drive retry logic | If you have only errno 107 |
| `fix_png_dcm_path_issue.py` | Path correction | If you have only path issues |
| `smart_image_loader.py` | Intelligent loader | For custom integration |

## Recommended Workflow

```
1. Try Solution 1 (one-line fix)
   ↓
2. If still failing → Run Solution 2 (diagnostic)
   ↓
3. Follow diagnostic recommendations
   OR
   Use Solution 3 (complete fix)
```

## Why These Errors Happen

### errno 107 (Transport endpoint not connected)
- **Cause**: Google Drive connection times out while reading files
- **Why**: Sequential reading of multiple files from mounted Drive
- **Solution**: Cache images locally before displaying

### FileNotFoundError (.dcm in .png folder)
- **Cause**: DICOM files converted to PNG but metadata not updated
- **Why**: Conversion script doesn't update metadata DataFrame
- **Solution**: Update extensions in metadata: `.dcm` → `.png`

## Prevention for Future

To prevent these issues in new notebooks:

```python
# After DICOM to PNG conversion, always do:

# 1. Update config
config.INBREAST_CONFIG['image_dir'] = 'images_png'

# 2. Update metadata paths
inbreast_metadata['image_path'] = inbreast_metadata['image_path'].str.replace('.dcm', '.png')

# 3. For visualization, use local caching
cache_dir = Path("/tmp/viz_cache")
# ... copy images to cache_dir before visualizing
```

## Testing Your Fix

After applying any fix, verify with:

```python
# Check first image loads successfully
from PIL import Image
image_dir = Path(config.INBREAST_PATH) / config.INBREAST_CONFIG["image_dir"]
test_path = image_dir / inbreast_metadata.iloc[0]['image_path']
print(f"Testing: {test_path}")
print(f"Exists: {test_path.exists()}")

if test_path.exists():
    img = Image.open(test_path)
    print(f"✓ Successfully loaded: {img.size}")
else:
    print("✗ File not found - fix not working")
```

## Quick Reference

| Error Message | Solution |
|---------------|----------|
| `errno 107` | Use `complete_visualization_fix.py` (has retry + cache) |
| `FileNotFoundError: .dcm` | Add: `str.replace('.dcm', '.png')` |
| Both errors | Use `complete_visualization_fix.py` (handles both) |
| Benign images fail | Use local caching (in complete fix) |
| Random failures | Use retry logic (in complete fix) |

## Support

All fixes have been tested and pushed to:
- **Repository**: https://github.com/dtobi59/mammography-multiobjective-optimization
- **Commits**:
  - `af60dec`: PNG/DCM path fixes
  - `60142bb`: errno 107 Drive retry logic

## TL;DR

**Just want it to work?**

Add this after loading metadata:
```python
inbreast_metadata['image_path'] = inbreast_metadata['image_path'].str.replace('.dcm', '.png', case=False)
```

**Want it to always work?**

Replace your visualization cell with code from `complete_visualization_fix.py`.
