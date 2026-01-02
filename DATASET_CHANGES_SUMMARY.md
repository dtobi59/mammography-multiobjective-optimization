# Dataset-Specific Changes Summary

## Overview

The implementation has been updated to handle **different dataset structures** for VinDr-Mammo and INbreast, as these datasets have different directory hierarchies, metadata formats, and labeling conventions.

---

## Key Changes

### 1. Dataset-Specific Parsers (`data/parsers.py`)

**New file:** Provides dataset-specific metadata parsers that convert native formats to a unified representation.

#### Features:
- ✅ **`VinDrMammoParser`**: Parses VinDr-Mammo CSV metadata
- ✅ **`INbreastParser`**: Parses INbreast CSV or XML metadata
- ✅ **`birads_to_binary_label()`**: Maps BI-RADS categories to binary labels
  - Handles standard categories (1-6)
  - Handles subcategories (4A, 4B, 4C)
  - Case-insensitive
- ✅ **`parse_dataset()`**: Unified interface for both datasets

#### BI-RADS Mapping:

| BI-RADS | Binary Label |
|---------|--------------|
| 1, 2, 3 | 0 (Benign) |
| 4, 4A, 4B, 4C, 5, 6 | 1 (Malignant) |
| 0 | Excluded |

### 2. Updated Configuration (`config.py`)

**Added:** Dataset-specific configuration dictionaries

```python
VINDR_CONFIG = {
    "metadata_file": "metadata.csv",
    "image_dir": "images",
    "image_id_col": "image_id",
    "patient_id_col": "study_id",
    "laterality_col": "laterality",
    "view_col": "view_position",
    "birads_col": "breast_birads",
    "image_extension": ".png",
}

INBREAST_CONFIG = {
    "metadata_file": "metadata.csv",
    "image_dir": "images",
    "metadata_format": "csv",
    "patient_id_col": "patient_id",
    "laterality_col": "laterality",
    "view_col": "view",
    "birads_col": "birads",
    "filename_col": "file_name",
}
```

### 3. Updated Data Loading (`data/__init__.py`)

**Added exports:**
- `VinDrMammoParser`
- `INbreastParser`
- `parse_dataset`
- `birads_to_binary_label`

### 4. Updated Optimization Runner (`optimization/nsga3_runner.py`)

**Changed:** `load_metadata()` function now uses dataset-specific parsers

**Before:**
```python
metadata = pd.read_csv(metadata_path)
```

**After:**
```python
metadata = parse_dataset(
    dataset_name="vindr",
    metadata_path=metadata_path,
    image_dir=image_dir,
    **parser_kwargs
)
```

### 5. Updated Evaluation Scripts

#### `evaluation/evaluate_source.py`
- Uses `VinDrMammoParser` via `load_metadata()`
- Loads from `VINDR_CONFIG`

#### `evaluation/evaluate_target.py`
- Uses `INbreastParser` via `load_metadata()`
- Loads from `INBREAST_CONFIG`
- Emphasizes zero-shot transfer (no fine-tuning, no threshold tuning)

### 6. New Documentation

#### `DATASET_SETUP_GUIDE.md` (NEW)
Comprehensive guide covering:
- Expected dataset structures
- Metadata format specifications
- BI-RADS mapping rules
- Configuration examples
- Custom parser creation
- Troubleshooting

#### Updated `README.md`
- References dataset-specific parsers
- Links to DATASET_SETUP_GUIDE.md
- Updated configuration section

### 7. New Tests (`test_parsers.py`)

**34 tests** covering:
- BI-RADS to binary label mapping (12 tests)
- VinDr-Mammo parser (11 tests)
- INbreast parser (5 tests)
- Unified representation (3 tests)
- Integration with Noisy OR (2 tests)
- Edge cases and error handling

**All tests passing (100%)**

---

## Unified Internal Representation

Both parsers output a **standardized DataFrame** with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | str | Unique image identifier |
| `patient_id` | str | Patient identifier |
| `breast_id` | str | Unique breast ID (`patient_id` + laterality) |
| `view` | str | "CC" or "MLO" |
| `label` | int | 0 (benign) or 1 (malignant) |
| `image_path` | str | Relative path to image |
| `birads_original` | str | Original BI-RADS category |

---

## Design Principles

### 1. Dataset-Specific Logic Only in Parsers

✅ **Parsers:** Handle dataset-specific formats, column names, file structures

✅ **All other code:** Operates on unified representation

❌ **No dataset-specific logic** in training, evaluation, or aggregation

### 2. Identical Downstream Processing

After parsing, **both datasets are processed identically**:

- Same training pipeline
- Same Noisy OR aggregation
- Same metric computation
- Same evaluation protocol

### 3. INbreast: Evaluation Only

✅ **INbreast is used strictly for inference and evaluation**

❌ No training on INbreast

❌ No fine-tuning on INbreast

❌ No threshold tuning on INbreast

✅ Pure zero-shot transfer

### 4. BI-RADS Consistency

Both datasets use BI-RADS assessments, but:

- **VinDr-Mammo:** Standard categories (1-6)
- **INbreast:** Includes subcategories (4A, 4B, 4C)

**Solution:** `birads_to_binary_label()` handles both formats consistently

---

## Files Modified

### New Files (3)
1. `data/parsers.py` - Dataset-specific parsers
2. `DATASET_SETUP_GUIDE.md` - Comprehensive setup guide
3. `test_parsers.py` - Parser test suite (34 tests)

### Modified Files (6)
1. `config.py` - Added VINDR_CONFIG and INBREAST_CONFIG
2. `data/__init__.py` - Export parser functions
3. `optimization/nsga3_runner.py` - Use parse_dataset()
4. `evaluation/evaluate_source.py` - Use VinDrMammoParser
5. `evaluation/evaluate_target.py` - Use INbreastParser
6. `README.md` - Updated data preparation section

### Documentation Files (1)
1. `DATASET_CHANGES_SUMMARY.md` - This file

---

## Backward Compatibility

### Breaking Changes

⚠️ **Old usage will NOT work:**

```python
# OLD (will fail)
metadata = pd.read_csv("metadata.csv")
```

✅ **New usage required:**

```python
# NEW (required)
from data.parsers import parse_dataset
metadata = parse_dataset(
    dataset_name="vindr",
    metadata_path="path/to/metadata.csv",
    image_dir="path/to/images",
    **config.VINDR_CONFIG
)
```

### Migration Path

1. Update `config.py` with dataset-specific configurations
2. Replace direct CSV loading with `parse_dataset()` or parser classes
3. Update any custom scripts to use unified representation

---

## Testing

### Test Coverage

| Test Suite | Tests | Status |
|-------------|-------|--------|
| Correctness Tests | 79 | ✅ PASS |
| Integration Test | 1 | ✅ PASS |
| **Parser Tests** | **34** | ✅ **PASS** |
| **Total** | **114** | ✅ **100%** |

### Run Tests

```bash
# All tests
python test_correctness.py  # 79 tests
python test_integration.py  # 1 test
python test_parsers.py      # 34 tests

# Quick check
python test_parsers.py && echo "Parsers OK"
```

---

## Benefits

### 1. Flexibility
- Easy to add new datasets
- Simple to handle different metadata formats
- Customizable column mappings

### 2. Consistency
- Unified representation ensures identical processing
- No dataset-specific bugs in downstream code
- Same Noisy OR aggregation for both datasets

### 3. Clarity
- Clear separation of concerns
- Dataset-specific logic isolated in parsers
- Easy to debug and maintain

### 4. Correctness
- BI-RADS subcategories handled properly
- Missing views handled identically
- Patient-wise splitting works for both datasets

---

## Future Enhancements

Potential future improvements:

1. **Additional datasets:** Create parsers for DDSM, CBIS-DDSM, etc.
2. **Auto-detection:** Automatically detect dataset type from metadata
3. **Validation:** More extensive metadata validation
4. **Visualization:** Tools to visualize dataset statistics
5. **Conversion tools:** Scripts to convert datasets to standard format

---

## Summary

The implementation now **correctly handles dataset-specific differences** while maintaining a **unified downstream pipeline**:

```
VinDr-Mammo → VinDrMammoParser → Unified DataFrame → Training/Evaluation
INbreast → INbreastParser → Unified DataFrame → Zero-Shot Evaluation
```

**Key principle:** Dataset-specific logic lives **only** in parsers. Everything else operates on the unified representation.

**Test status:** All 114 tests passing (100%)

**Ready for use:** ✅
