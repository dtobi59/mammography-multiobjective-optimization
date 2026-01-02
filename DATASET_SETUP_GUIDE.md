# Dataset Setup Guide

## Overview

This implementation supports **two datasets with different structures**:

1. **VinDr-Mammo** (Source domain) - used for training and validation
2. **INbreast** (Target domain) - used for zero-shot evaluation only

**Important:** The datasets have different directory structures, metadata formats, and naming conventions. Dataset-specific parsers handle these differences and convert both to a unified internal representation.

---

## Unified Internal Representation

Both parsers output a standardized DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | str | Unique image identifier |
| `patient_id` | str | Patient identifier (for patient-wise splitting) |
| `breast_id` | str | Unique breast identifier (`patient_id` + laterality) |
| `view` | str | View type: "CC" or "MLO" |
| `label` | int | Binary label: 0 (benign) or 1 (malignant/suspicious) |
| `image_path` | str | Relative path to image file |
| `birads_original` | str | Original BI-RADS category (for reference) |

**All downstream code operates on this unified representation** - no dataset-specific logic is allowed beyond loading and label mapping.

---

## BI-RADS to Binary Label Mapping

Both datasets use BI-RADS assessments that must be mapped to binary labels:

### Mapping Rules

| BI-RADS | Description | Binary Label |
|---------|-------------|--------------|
| 1 | Negative | **0** (Benign) |
| 2 | Benign | **0** (Benign) |
| 3 | Probably benign | **0** (Benign) |
| 4 | Suspicious | **1** (Malignant) |
| 4A | Low suspicion | **1** (Malignant) |
| 4B | Moderate suspicion | **1** (Malignant) |
| 4C | High suspicion | **1** (Malignant) |
| 5 | Highly suspicious | **1** (Malignant) |
| 6 | Biopsy-proven malignancy | **1** (Malignant) |
| 0 | Incomplete (needs additional imaging) | **Excluded** |

**Note:** BI-RADS subcategories (4A, 4B, 4C) in INbreast are all mapped to label 1.

---

## VinDr-Mammo Dataset Setup

### Expected Structure

```
vindr_mammo/
├── metadata.csv              # Metadata file
└── images/                   # Directory containing PNG images
    ├── image001.png
    ├── image002.png
    └── ...
```

### Metadata Format

VinDr-Mammo metadata should be a CSV file with these columns (actual names may vary):

| Column | Description | Example |
|--------|-------------|---------|
| `image_id` | Image filename (without extension) | "image001" |
| `study_id` | Study/patient identifier | "P12345" |
| `laterality` | Breast side: L or R | "L" |
| `view_position` | View type | "CC", "MLO" |
| `breast_birads` | BI-RADS assessment | "2", "4A", "5" |

**Example CSV:**

```csv
image_id,study_id,laterality,view_position,breast_birads
img001,P001,L,CC,2
img002,P001,L,MLO,2
img003,P001,R,CC,4A
img004,P001,R,MLO,4A
img005,P002,L,CC,5
```

### Configuration

Edit `config.py` to set:

```python
VINDR_MAMMO_PATH = "/path/to/vindr_mammo"

VINDR_CONFIG = {
    "metadata_file": "metadata.csv",
    "image_dir": "images",
    "image_id_col": "image_id",        # Column name for image ID
    "patient_id_col": "study_id",      # Column name for patient ID
    "laterality_col": "laterality",    # Column name for laterality (L/R)
    "view_col": "view_position",       # Column name for view type
    "birads_col": "breast_birads",     # Column name for BI-RADS
    "image_extension": ".png",         # Image file extension
}
```

**Adjust column names** to match your actual CSV header.

### Images

- **Format:** PNG (converted from DICOM)
- **Naming:** Filenames should match `image_id` + `image_extension`
- **Size:** Any size (will be resized to 224×224)
- **Channels:** Grayscale or RGB

---

## INbreast Dataset Setup

### Expected Structure

```
inbreast/
├── metadata.csv              # Metadata file (or metadata.xml)
└── images/                   # Directory containing images
    ├── patient01_L_CC.png
    ├── patient01_L_MLO.png
    └── ...
```

### Metadata Format

INbreast metadata can be CSV or XML. For CSV format:

| Column | Description | Example |
|--------|-------------|---------|
| `patient_id` | Patient identifier | "patient01" |
| `file_name` | Image filename | "patient01_L_CC.png" |
| `laterality` | Breast side: L or R | "L" |
| `view` | View type | "CC", "MLO" |
| `birads` | BI-RADS assessment (may include subcategories) | "3", "4B", "5" |

**Example CSV:**

```csv
patient_id,file_name,laterality,view,birads
patient01,patient01_L_CC.png,L,CC,3
patient01,patient01_L_MLO.png,L,MLO,3
patient02,patient02_R_CC.png,R,CC,4B
patient02,patient02_R_MLO.png,R,MLO,4B
```

### Configuration

Edit `config.py` to set:

```python
INBREAST_PATH = "/path/to/inbreast"

INBREAST_CONFIG = {
    "metadata_file": "metadata.csv",   # or "metadata.xml"
    "image_dir": "images",
    "metadata_format": "csv",          # "csv" or "xml"
    "patient_id_col": "patient_id",
    "laterality_col": "laterality",
    "view_col": "view",
    "birads_col": "birads",
    "filename_col": "file_name",
}
```

**For XML format:** The parser will attempt to extract fields from XML. You may need to adjust `data/parsers.py:INbreastParser._parse_xml()` based on your XML schema.

### Images

- **Format:** PNG or other image formats supported by PIL
- **Naming:** Can be any naming scheme (specified in metadata `file_name`)
- **Size:** Any size (will be resized to 224×224)
- **Channels:** Grayscale or RGB

---

## Usage

### 1. Verify Setup

After configuring datasets, run:

```bash
python test_setup.py \
  --vindr_metadata path/to/vindr_mammo/metadata.csv \
  --vindr_images path/to/vindr_mammo/images \
  --inbreast_metadata path/to/inbreast/metadata.csv \
  --inbreast_images path/to/inbreast/images
```

This will check:
- Metadata files are readable
- Required columns present
- Images exist and are loadable
- BI-RADS mapping works
- Patient-wise grouping correct

### 2. Test Parsers

Test dataset-specific parsers:

```bash
python test_parsers.py
```

This verifies:
- BI-RADS to binary label mapping
- VinDr-Mammo parser produces correct format
- INbreast parser produces correct format
- Both output unified representation
- Integration with Noisy OR aggregation

### 3. Run Optimization

Train on VinDr-Mammo:

```bash
python optimization/nsga3_runner.py
```

The runner will:
- Load VinDr-Mammo using `VinDrMammoParser`
- Create patient-wise 80/20 split
- Train models on unified representation
- Optimize 4 objectives using NSGA-III

### 4. Evaluate on Source

Evaluate on VinDr-Mammo validation set:

```bash
python evaluation/evaluate_source.py \
  --checkpoint checkpoints/eval_X/best_checkpoint.pt \
  --hyperparameters solution_Y_config.json
```

### 5. Zero-Shot Evaluation on Target

Evaluate on INbreast (zero-shot):

```bash
python evaluation/evaluate_target.py \
  --checkpoint checkpoints/eval_X/best_checkpoint.pt \
  --threshold 0.45 \
  --hyperparameters solution_Y_config.json
```

**Important:**
- No fine-tuning on INbreast
- No threshold tuning on INbreast
- Threshold transferred from source validation
- Pure zero-shot transfer evaluation

---

## Breast-Level Aggregation

Breast-level aggregation using **Noisy OR** is applied **identically** to both datasets after parsing:

```python
p_breast = 1 - (1 - p_CC) * (1 - p_MLO)
```

**Key points:**
- Applied to unified representation (no dataset-specific logic)
- Missing views handled identically (probability = 0)
- Multiple views of same type: max probability used
- Same aggregation function for both VinDr-Mammo and INbreast

---

## Common Issues

### Issue: "Missing required columns"

**Cause:** Column names in your CSV don't match configuration

**Solution:** Update `config.py` to match your actual column names. For example:

```python
VINDR_CONFIG = {
    # ... other settings ...
    "patient_id_col": "StudyInstanceUID",  # Use your actual column name
    "laterality_col": "ImageLaterality",
    # ... etc ...
}
```

### Issue: "Unknown BI-RADS category"

**Cause:** BI-RADS value not recognized (e.g., "N/A", "Unknown")

**Solution:** Clean your metadata to remove or map invalid BI-RADS values. The parser will skip images with invalid BI-RADS.

### Issue: "Image not found"

**Cause:** `image_path` in metadata doesn't match actual filenames

**Solution:**
1. Check `image_extension` in config matches your files
2. Verify `image_id` in metadata matches filenames
3. Ensure images are in correct directory

### Issue: Different metadata format

**Cause:** Your dataset uses different CSV structure or XML schema

**Solution:**
1. For CSV: Update column mappings in `config.py`
2. For XML: Modify `data/parsers.py:INbreastParser._parse_xml()` to match your schema
3. Or create a custom parser by subclassing the base parsers

---

## Creating Custom Parsers

If your dataset has a unique format, create a custom parser:

```python
from data.parsers import VinDrMammoParser
import pandas as pd

class CustomParser(VinDrMammoParser):
    def parse(self) -> pd.DataFrame:
        # Load your custom format
        df = self._load_custom_format()

        # Convert to standardized format
        standardized = pd.DataFrame()
        standardized["image_id"] = df["your_image_id_column"]
        standardized["patient_id"] = df["your_patient_id_column"]

        # Construct breast_id
        laterality = df["your_laterality_column"]
        standardized["breast_id"] = standardized["patient_id"] + "_" + laterality

        # Normalize view
        standardized["view"] = df["your_view_column"].map(lambda x: "CC" if "CC" in x else "MLO")

        # Map BI-RADS to binary label
        from data.parsers import birads_to_binary_label
        standardized["label"] = df["your_birads_column"].apply(birads_to_binary_label)

        # Other required columns
        standardized["image_path"] = df["your_filename_column"]
        standardized["birads_original"] = df["your_birads_column"]

        return standardized
```

**Required columns in output:**
- `image_id`, `patient_id`, `breast_id`, `view`, `label`, `image_path`

---

## Validation Checklist

Before running optimization, verify:

- [ ] VinDr-Mammo metadata loaded successfully
- [ ] INbreast metadata loaded successfully
- [ ] All images exist and are loadable
- [ ] BI-RADS values mapped to binary labels (0/1)
- [ ] Patient IDs are unique and consistent
- [ ] Breast IDs combine patient_id + laterality
- [ ] Views normalized to "CC" or "MLO"
- [ ] Image paths are correct
- [ ] No patient overlap between train/validation splits
- [ ] Both datasets output identical schema

Run `python test_parsers.py` to verify all of the above.

---

## Summary

| Aspect | VinDr-Mammo | INbreast |
|--------|-------------|----------|
| **Usage** | Training + Validation | Zero-shot evaluation only |
| **Metadata format** | CSV (dataset-specific columns) | CSV or XML (different structure) |
| **BI-RADS** | Standard (1-6) | Includes subcategories (4A, 4B, 4C) |
| **Parser** | `VinDrMammoParser` | `INbreastParser` |
| **Output** | Unified DataFrame | Unified DataFrame (identical schema) |
| **Aggregation** | Noisy OR (identical implementation) | Noisy OR (identical implementation) |
| **Dataset-specific logic** | **Only in parser** | **Only in parser** |
| **Downstream code** | **Unified representation** | **Unified representation** |

**Key principle:** Dataset-specific differences are handled **only** in the parsers. All training, evaluation, and aggregation logic operates on the unified representation.
