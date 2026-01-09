# Breast-Level Classification Fix

## Issue Identified

The original breast-level aggregation code in `utils/noisy_or.py` used an **incorrect method** for determining breast-level ground truth labels:

```python
# INCORRECT (line 79)
breast_label = group["label"].iloc[0]  # Takes first image's label
```

### Why This Was Wrong

1. **Arbitrary Selection**: Taking `.iloc[0]` arbitrarily selects the first image's label
2. **Assumes Consistency**: Assumes all views of the same breast have identical labels
3. **Violates Clinical Logic**: Different views (CC, MLO) can have different BI-RADS assessments
4. **Potential for Errors**: Could misclassify breasts if views have different labels

## The Fix

Changed to use `max()` which follows clinical best practice:

```python
# CORRECT (new implementation)
breast_label = int(group["label"].max())  # If ANY view is positive, breast is positive
```

### Why This Is Correct

1. **Clinical Logic**: If **any** view shows malignancy (label=1), the breast should be classified as malignant
2. **Conservative Approach**: Doesn't miss positive cases
3. **Handles All Cases**: Works correctly whether views have consistent or inconsistent labels
4. **Standard Practice**: Aligns with radiological diagnostic principles

## Medical Rationale

In mammography:
- A breast is imaged from multiple views (typically CC and MLO)
- Each view shows different aspects of the breast tissue
- A lesion may be more visible in one view than another
- If **any view** indicates malignancy, the breast is considered suspicious
- This follows the principle: "positive in any view → positive breast"

## Mathematical Justification

For predictions, the code uses **Noisy OR** aggregation:
```
p_breast = 1 - ∏(1 - p_view)
```

This is equivalent to "if any view predicts positive, breast is positive" in probabilistic form.

For ground truth, we should use the same logic:
```
label_breast = max(label_view)
```

This ensures consistency between prediction aggregation and ground truth aggregation.

## Verification

A verification script (`verify_breast_labels.py`) was added to check if datasets have consistent labels across views:

```bash
python verify_breast_labels.py
```

This script:
- Checks each breast to see if all views have the same label
- Reports any inconsistencies found
- Provides statistics on label consistency

## Impact

### Before Fix
- **Risk**: If CC view was labeled 0 and MLO view was labeled 1, and CC was first, the breast would be incorrectly labeled as benign
- **Severity**: Could cause false negatives in evaluation metrics
- **Likelihood**: Depends on dataset - some may have consistent labels, others may not

### After Fix
- **Guarantee**: Breast is labeled positive if ANY view is positive
- **Accuracy**: Follows clinical diagnostic logic
- **Robustness**: Handles both consistent and inconsistent cases correctly

## Files Changed

1. **`utils/noisy_or.py`** (line 79-80)
   - Changed from `.iloc[0]` to `.max()`
   - Added explanatory comments

2. **`verify_breast_labels.py`** (NEW)
   - Verification script to check label consistency
   - Reports any breasts with mixed labels
   - Provides recommendations

## Testing

To verify the fix works correctly:

1. Run the verification script:
   ```bash
   python verify_breast_labels.py
   ```

2. Check evaluation results before/after:
   - If dataset had inconsistent labels, metrics should change
   - If dataset had consistent labels, no change expected

3. Visual inspection:
   ```python
   # Check a specific breast
   breast_data = metadata[metadata['breast_id'] == 'patient_123_L']
   print(breast_data[['image_id', 'view', 'label']])
   ```

## Recommendations

1. **Always run `verify_breast_labels.py`** on new datasets to check consistency
2. **Document** if your dataset has view-level or breast-level labels
3. **Re-evaluate** existing results if they used the old implementation
4. **Consider** whether your specific use case requires different logic

## Clinical Context

This fix aligns with standard radiological practice:

- **Screening**: Any suspicious finding warrants further investigation
- **Diagnosis**: Multiple views increase sensitivity
- **Treatment**: Even if one view is clear, a positive in another view is significant

The `max()` approach ensures our evaluation metrics reflect this clinical reality.

## Backwards Compatibility

### No Breaking Changes
- Function signature unchanged
- API remains the same
- Only the internal logic improved

### Behavior Change
- If all views have the same label: **No change**
- If views have different labels: **Now uses max (correct)**

Since the change makes the code **more correct**, any differences in metrics are **improvements** not regressions.

## References

1. **Noisy OR Logic**: Widely used in multi-instance learning
2. **Clinical Practice**: Standard mammography interpretation guidelines
3. **Medical Literature**: Multiple views increase diagnostic sensitivity

---

**Summary**: This fix ensures breast-level classification follows clinical logic where a positive finding in any view classifies the entire breast as positive, which is medically sound and mathematically consistent with the Noisy OR prediction aggregation.
