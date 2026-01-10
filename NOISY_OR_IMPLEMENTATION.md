# Breast-Level Evaluation: Noisy OR Implementation

## Executive Summary

This document shows the **exact implementation** of breast-level evaluation using the generalized Noisy OR formula. All implementations are verified with automated unit tests.

---

## 1. breast_id Definition

**File:** `data/parsers.py`
**Line:** 438

### Code:
```python
standardized["breast_id"] = standardized["patient_id"] + "_" + laterality
```

### Context (lines 430-438):
```python
def _parse_csv_like(self, df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame to standardized format."""
    standardized = pd.DataFrame()

    standardized["image_id"] = df["filename"].apply(lambda x: Path(x).stem)
    standardized["patient_id"] = df["patient_id"].astype(str)

    laterality = df["laterality"].astype(str).str.upper().str.strip()
    standardized["breast_id"] = standardized["patient_id"] + "_" + laterality
```

### Formula:
```
breast_id = patient_id + "_" + laterality
```

### Examples:
- Patient `P001`, Right breast → `P001_R`
- Patient `P001`, Left breast → `P001_L`
- Patient `48575a27b7c992427041a82fa750d3fa`, Right breast → `48575a27b7c992427041a82fa750d3fa_R`

### Verification:
✅ Tested in `test_noisy_or_aggregation.py::test_breast_id_definition`

---

## 2. Grouping Logic

**File:** `utils/noisy_or.py`
**Line:** 53

### Code:
```python
breast_groups = metadata.groupby("breast_id")
```

### Full Context (lines 52-58):
```python
# Group by breast_id
breast_groups = metadata.groupby("breast_id")

breast_predictions = []
breast_labels = []

for breast_id, group in breast_groups:
```

### Data Structure:
- **Input:** `metadata` DataFrame with columns `[image_id, patient_id, breast_id, view, label]`
- **Grouping key:** `breast_id` (e.g., `P001_R`, `P001_L`)
- **Output:** Pandas `GroupBy` object that iterates over (breast_id, DataFrame) tuples

### Example:
```
Original metadata:
image_id    patient_id  breast_id  view  label
img_cc      P001        P001_R     CC    1
img_mlo     P001        P001_R     MLO   1

After groupby("breast_id"):
Group 1: breast_id = "P001_R"
  DataFrame with 2 rows (img_cc and img_mlo)
```

### Verification:
✅ Tested in `test_noisy_or_aggregation.py::test_groupby_multiple_breasts`

---

## 3. Generalized Noisy OR Formula

**File:** `utils/noisy_or.py`
**Lines:** 26-83

### Mathematical Formula:
```
For m images of the same breast:
    p_breast = 1 - ∏_{j=1}^{m} (1 - p_j)

Where:
- p_j is the predicted probability for image j
- ∏ denotes product (multiplication)
- m is the number of views for this breast
```

### Implementation:
```python
def aggregate_to_breast_level(
    image_predictions: Dict[str, float],
    metadata: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate image-level predictions to breast-level using generalized Noisy OR.

    Generalized Noisy OR formula for m images:
        p_breast = 1 - ∏_{j=1}^{m} (1 - p_j)

    Special cases:
        - m = 1: p_breast = p_1 (single view)
        - m = 2: p_breast = 1 - (1 - p_1)(1 - p_2) (standard Noisy OR)
        - m > 2: generalized formula applies
    """
    # Group by breast_id
    breast_groups = metadata.groupby("breast_id")

    breast_predictions = []
    breast_labels = []

    for breast_id, group in breast_groups:
        # Collect all available image predictions for this breast
        image_probs = []
        for img_id in group["image_id"]:
            if img_id in image_predictions:
                image_probs.append(image_predictions[img_id])

        # Apply generalized Noisy OR
        if len(image_probs) == 0:
            # No predictions available, default to 0
            breast_prob = 0.0
        elif len(image_probs) == 1:
            # Single view: p_breast = p_1
            breast_prob = image_probs[0]
        else:
            # Multiple views: p_breast = 1 - ∏(1 - p_j)
            breast_prob = 1.0 - np.prod([1.0 - p for p in image_probs])

        breast_predictions.append(breast_prob)

        # Ground truth label: max across all views
        breast_label = int(group["label"].max())
        breast_labels.append(breast_label)

    return np.array(breast_predictions), np.array(breast_labels)
```

### Key Implementation Details:

#### Case 1: No predictions available (m = 0)
**Lines:** 66-68
```python
if len(image_probs) == 0:
    breast_prob = 0.0
```

#### Case 2: Single view (m = 1)
**Lines:** 69-71
```python
elif len(image_probs) == 1:
    # Single view: p_breast = p_1
    breast_prob = image_probs[0]
```

**Special Case Confirmed:** When only one view exists, `p_breast = p_1` (no aggregation needed)

#### Case 3: Multiple views (m ≥ 2)
**Lines:** 72-74
```python
else:
    # Multiple views: p_breast = 1 - ∏(1 - p_j)
    breast_prob = 1.0 - np.prod([1.0 - p for p in image_probs])
```

---

## 4. Label Aggregation

**File:** `utils/noisy_or.py`
**Line:** 80

### Code:
```python
breast_label = int(group["label"].max())
```

### Logic:
- **Formula:** `breast_label = max(label_1, label_2, ..., label_m)`
- **Meaning:** If ANY view shows malignancy (label=1), the breast is considered malignant
- **Clinical Rationale:** Conservative approach that prioritizes sensitivity over specificity

### Example:
```
Views for breast P001_R:
  CC view:  label = 0 (benign)
  MLO view: label = 1 (malignant)

breast_label = max(0, 1) = 1 (malignant)
```

### Verification:
✅ Tested in `test_noisy_or_aggregation.py::test_label_aggregation_max`

---

## 5. Numerical Examples with Verification

### Example 1: Single View

**Input:**
- Breast: `P001_R`
- Views: CC only
- Prediction: `p_CC = 0.7`

**Computation:**
```
m = 1
p_breast = p_1 = 0.7
```

**Test Result:** ✅ PASSED
**File:** `test_noisy_or_aggregation.py::test_single_view_case`

---

### Example 2: Two Views (Standard Noisy OR)

**Input:**
- Breast: `P001_R`
- Views: CC and MLO
- Predictions: `p_CC = 0.6`, `p_MLO = 0.8`

**Computation:**
```
m = 2
p_breast = 1 - (1 - p_CC)(1 - p_MLO)
         = 1 - (1 - 0.6)(1 - 0.8)
         = 1 - (0.4)(0.2)
         = 1 - 0.08
         = 0.92
```

**Test Result:** ✅ PASSED
**File:** `test_noisy_or_aggregation.py::test_two_views_case`

---

### Example 3: Three Views (Generalized)

**Input:**
- Breast: `P001_R`
- Views: CC, MLO, CC (repeat imaging)
- Predictions: `p_1 = 0.5`, `p_2 = 0.6`, `p_3 = 0.7`

**Computation:**
```
m = 3
p_breast = 1 - (1 - p_1)(1 - p_2)(1 - p_3)
         = 1 - (1 - 0.5)(1 - 0.6)(1 - 0.7)
         = 1 - (0.5)(0.4)(0.3)
         = 1 - 0.06
         = 0.94
```

**Test Result:** ✅ PASSED
**File:** `test_noisy_or_aggregation.py::test_three_views_generalized`

---

### Example 4: Realistic Clinical Scenario

**Scenario:** Patient P001 with bilateral findings

**Input:**
- Right breast (`P001_R`): CC view = 0.65, MLO view = 0.72
- Left breast (`P001_L`): CC view only = 0.15

**Computation:**
```
P001_R (two views):
  p_breast = 1 - (1 - 0.65)(1 - 0.72)
           = 1 - (0.35)(0.28)
           = 1 - 0.098
           = 0.902

P001_L (one view):
  p_breast = 0.15
```

**Test Result:** ✅ PASSED
**File:** `test_noisy_or_aggregation.py::test_numerical_example_from_paper`

---

## 6. Test Suite Summary

**Test File:** `test_noisy_or_aggregation.py`

### All Tests:
1. ✅ `test_breast_id_definition` - Verifies `breast_id = patient_id + "_" + laterality`
2. ✅ `test_single_view_case` - Confirms `m=1 → p_breast = p_1`
3. ✅ `test_two_views_case` - Tests standard Noisy OR with two views
4. ✅ `test_three_views_generalized` - Tests generalized formula with m>2
5. ✅ `test_groupby_multiple_breasts` - Verifies groupby logic with multiple patients
6. ✅ `test_label_aggregation_max` - Confirms max() aggregation for labels
7. ✅ `test_noisy_or_function_two_views` - Tests standalone noisy_or_aggregation()
8. ✅ `test_extreme_values` - Tests edge cases (0.0, 1.0 probabilities)
9. ✅ `test_numerical_example_from_paper` - Realistic clinical example

### Run Command:
```bash
python test_noisy_or_aggregation.py
```

### Results:
```
Tests run:     9
Successes:     9
Failures:      0
Errors:        0

[OK] ALL TESTS PASSED - Noisy OR implementation verified
```

---

## 7. Formula Verification Table

| Views (m) | Input Probabilities | Formula | Expected Output | Actual Output | Status |
|-----------|-------------------|---------|-----------------|---------------|--------|
| 1 | p₁=0.7 | p₁ | 0.700000 | 0.700000 | ✅ |
| 2 | p₁=0.6, p₂=0.8 | 1-(1-p₁)(1-p₂) | 0.920000 | 0.920000 | ✅ |
| 3 | p₁=0.5, p₂=0.6, p₃=0.7 | 1-(1-p₁)(1-p₂)(1-p₃) | 0.940000 | 0.940000 | ✅ |
| 2 | p₁=0.65, p₂=0.72 | 1-(1-p₁)(1-p₂) | 0.902000 | 0.902000 | ✅ |
| 2 | p₁=0.0, p₂=0.0 | 1-(1-p₁)(1-p₂) | 0.000000 | 0.000000 | ✅ |
| 2 | p₁=1.0, p₂=1.0 | 1-(1-p₁)(1-p₂) | 1.000000 | 1.000000 | ✅ |

---

## 8. Code Locations Summary

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **breast_id definition** | `data/parsers.py` | 438 | `patient_id + "_" + laterality` |
| **Grouping logic** | `utils/noisy_or.py` | 53 | `metadata.groupby("breast_id")` |
| **Noisy OR aggregation** | `utils/noisy_or.py` | 26-83 | Full implementation |
| **Single view case** | `utils/noisy_or.py` | 69-71 | `p_breast = p_1` |
| **Multiple views case** | `utils/noisy_or.py` | 72-74 | `p_breast = 1 - ∏(1-p_j)` |
| **Label aggregation** | `utils/noisy_or.py` | 80 | `max(labels)` |
| **Test suite** | `test_noisy_or_aggregation.py` | 1-369 | 9 comprehensive tests |

---

## 9. Mathematical Properties

### Property 1: Monotonicity
If any `p_j` increases, `p_breast` increases (never decreases).

**Proof:**
```
∂p_breast/∂p_j = ∏_{k≠j}(1 - p_k) ≥ 0
```

### Property 2: Boundary Conditions
- If all `p_j = 0`, then `p_breast = 0`
- If any `p_j = 1`, then `p_breast = 1`

**Verified:** ✅ `test_extreme_values` confirms both conditions

### Property 3: Commutativity
Order of views doesn't matter: `p(p₁, p₂) = p(p₂, p₁)`

**Proof:** Product operation is commutative.

### Property 4: Special Case Equivalence
When `m=1`, Noisy OR reduces to identity: `p_breast = p_1`

**Verified:** ✅ `test_single_view_case`

---

## 10. Implementation Guarantees

1. ✅ **Correct formula**: Generalized Noisy OR implemented exactly as specified
2. ✅ **Special case handling**: Single view case returns `p_1` without modification
3. ✅ **Numerical accuracy**: All test cases match expected values to 6 decimal places
4. ✅ **Grouping correctness**: Pandas groupby correctly partitions by `breast_id`
5. ✅ **Label aggregation**: max() ensures any positive view makes breast positive
6. ✅ **Edge cases**: Handles 0.0, 1.0, and missing predictions correctly

---

## Verification Commands

### Run Noisy OR tests:
```bash
python test_noisy_or_aggregation.py
```

### Run all patient leakage tests:
```bash
python test_patient_leakage.py
```

### Check INbreast grouping:
```bash
python check_label_consistency.py
```

---

## References

- **Noisy OR implementation:** `utils/noisy_or.py:26-83`
- **breast_id construction:** `data/parsers.py:438`
- **Test suite:** `test_noisy_or_aggregation.py`
- **Patient leakage proof:** `PROOF_NO_PATIENT_LEAKAGE.md`

**Conclusion:** The breast-level evaluation using generalized Noisy OR is **correctly implemented** and **numerically verified** across all test cases.
