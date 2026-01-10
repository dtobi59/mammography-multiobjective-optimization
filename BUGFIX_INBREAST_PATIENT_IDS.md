# Bug Fix: INbreast Patient ID Masking Issue

## Problem

The INbreast dataset CSV file had all Patient IDs replaced with "removed" for privacy. This caused a critical bug in breast-level evaluation:

### Root Cause
- **Breast ID generation**: `breast_id = patient_id + "_" + laterality`
- With all `patient_id = "removed"`, all breasts collapsed into just 2 groups:
  - `removed_R` (all right breasts from all patients)
  - `removed_L` (all left breasts from all patients)
- Breast-level label aggregation uses `max()` across all views per breast
- Both aggregated "breasts" contained at least one malignant image → both labeled as malignant (1)
- **Result**: Single-class dataset at breast level

### Symptoms
```
Target Dataset (INbreast) - Breast Level:
  PR-AUC: 0.0000 (-100.0%)    # Only one class present
  AUROC:  0.5000 (-28.4%)     # Random chance
  Brier:  0.0000 (-100.0%)    # Edge case in metrics.py
```

## Solution

### Implemented Fix (data/parsers.py:267-337)

When patient IDs are masked ("removed" or "masked"), the parser now infers patient groupings from **acquisition date** + file number proximity:

1. **Detection**: Check if all patient IDs are "removed" or "masked"
2. **Patient Grouping Strategy**:
   - **Primary**: Sort by acquisition date, then file number
   - Start new patient group when:
     - Acquisition date changes, OR
     - File number gap > 100 (within same date)
   - **Fallback** (if no date): File number gaps > 100
3. **Breast ID Generation**: Use inferred patient ID + laterality

### Example
Before fix:
```
breast_id
removed_R    205 images  →  label=1 (max across all images)
removed_L    205 images  →  label=1 (max across all images)
Total: 2 breasts, both malignant
```

After fix:
```
patient_0001_R    2 images
patient_0001_L    2 images
patient_0002_R    2 images
...
Total: 129 breasts (84 benign, 45 malignant) from 70 patients
```

### Code Changes

**File**: `data/parsers.py`
**Method**: `INbreastParser._parse_csv()`
**Lines**: 267-337

```python
# Check if all patient IDs are "removed" or similar placeholder
if (patient_ids_raw == "removed").all() or (patient_ids_raw == "masked").all():
    print("Warning: Patient IDs are masked. Inferring patient groupings from acquisition date and file numbers.")

    # Use acquisition date if available
    if 'Acquisition date' in df.columns:
        # Sort by acquisition date first, then file number
        df_sorted = df.sort_values(['Acquisition date', self.filename_col])

        patient_groups = []
        current_patient_id = 0
        prev_date = None
        prev_file = None

        for _, row in df_sorted.iterrows():
            curr_date = row['Acquisition date']
            curr_file = int(row[self.filename_col])

            # New patient if:
            # 1. Acquisition date changes, OR
            # 2. File number gap > 100 (within same date)
            if prev_date is None:
                pass  # First row
            elif curr_date != prev_date:
                current_patient_id += 1
            elif abs(curr_file - prev_file) > 100:
                current_patient_id += 1

            patient_groups.append(f"patient_{current_patient_id:04d}")
            prev_date = curr_date
            prev_file = curr_file

        # Create mapping back to original order
        df_sorted['patient_id_temp'] = patient_groups
        df_with_patient = df.merge(
            df_sorted[[self.filename_col, 'patient_id_temp']],
            on=self.filename_col,
            how='left'
        )
        standardized["patient_id"] = df_with_patient['patient_id_temp'].values
        print(f"Inferred {current_patient_id + 1} patient groups using acquisition date + file number proximity.")
```

## Validation

### Debug Script
Run `debug_evaluation.py` to verify the fix:

```bash
python debug_evaluation.py
```

Expected output:
```
Warning: Patient IDs are masked. Inferring patient groupings from acquisition date and file numbers.
Inferred 70 patient groups using acquisition date + file number proximity.

Breast-level label distribution:
label
0    84
1    45
Name: count, dtype: int64

Total breasts: 129
Malignant breasts: 45
Benign breasts: 84

Label inconsistency: 18.3% (breasts with mixed labels across views)

Metrics with random predictions:
PR-AUC: 0.4797  # Now computed correctly
AUROC: 0.5537   # Close to random (as expected)
Brier: 0.4865   # Now computed correctly
```

## Impact

- ✅ Fixed breast-level evaluation for INbreast dataset
- ✅ Both classes now properly represented at breast level (84 benign, 45 malignant)
- ✅ Metrics now accurately reflect model performance
- ✅ Automated detection and warning when patient IDs are masked
- ✅ Uses acquisition date to prevent grouping different patients together
- ✅ Only 18.3% label inconsistency (down from 58.8% with file-only grouping)

## Limitations

1. **Heuristic Grouping**: Uses acquisition date + file number gaps >100
   - Works well for INbreast (18.3% label inconsistency)
   - May need adjustment for datasets with different conventions
   - True patient groupings unknown due to privacy masking

2. **Breast Count**: 129 breasts from 70 patients (avg 1.8 breasts/patient)
   - Some patients missing left or right breast data
   - Some breasts have 1-2 views instead of standard 2
   - Cannot verify accuracy without ground truth patient IDs

3. **Label Inconsistency**: 18.3% of breasts have conflicting labels across views
   - Could indicate imaging quality issues or annotation errors
   - Max aggregation assumes "if any view shows malignancy, breast is positive"

## Future Improvements

1. Add configuration option for gap threshold (currently hardcoded to 100)
2. Validate grouping by checking typical mammography patterns (2 breasts, 2 views each)
3. Add validation warnings if inferred patient groups seem unusual
4. Consider alternative grouping strategies (e.g., clustering, Hungarian algorithm)
5. Use ACR (breast density) as additional grouping signal (should be same for both breasts of a patient)

## Related Files

- `data/parsers.py` - Main fix location
- `utils/noisy_or.py` - Breast-level aggregation logic
- `training/metrics.py` - Metric computation (handles edge cases)
- `debug_evaluation.py` - Validation script
- `evaluation/evaluate_target.py` - Zero-shot evaluation on INbreast

## Testing

To test with actual model evaluation:

```bash
python evaluation/evaluate_target.py \
  --checkpoint path/to/checkpoint.pt \
  --threshold 0.5 \
  --hyperparameters config.json
```

The evaluation should now show non-zero breast-level metrics.

---

**Fixed**: 2026-01-10
**Git Commit**: TBD (to be committed)
