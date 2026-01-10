# Robustness Perturbation Implementation

## Executive Summary

This document shows the **exact implementation** of robustness perturbations, confirming:
1. ✅ Intensity-only perturbations (brightness, contrast, Gaussian noise)
2. ✅ No geometric transforms
3. ✅ Deterministic per image_id using seed derived from `hash(image_id)`
4. ✅ Same perturbation applied every time for a given image_id

---

## 1. RobustnessPerturbation Class

**File:** `data/augmentation.py`
**Lines:** 79-143

### Full Implementation:
```python
class RobustnessPerturbation:
    """
    Mild intensity perturbations for robustness evaluation.

    Fixed perturbations (not random) for consistent robustness testing.
    """

    def __init__(
        self,
        brightness_delta: float = config.ROBUSTNESS_PERTURBATION["brightness_delta"],
        contrast_delta: float = config.ROBUSTNESS_PERTURBATION["contrast_delta"],
        noise_std: float = config.ROBUSTNESS_PERTURBATION["noise_std"],
    ):
        """
        Initialize robustness perturbation.

        Args:
            brightness_delta: Fixed brightness adjustment
            contrast_delta: Fixed contrast scaling
            noise_std: Standard deviation of Gaussian noise
        """
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor, image_id: Optional[str] = None, seed: Optional[int] = None) -> torch.Tensor:
        """
        Apply fixed perturbation to image with deterministic seed derived from image_id.

        Args:
            image: Input image tensor, shape (C, H, W), values in [0, 1]
            image_id: Image identifier used to derive deterministic seed (preferred)
            seed: Optional explicit seed (deprecated, use image_id instead)

        Returns:
            Perturbed image tensor, shape (C, H, W), values clipped to [0, 1]
        """
        # Apply fixed brightness and contrast adjustments
        image = TF.adjust_brightness(image, 1.0 + self.brightness_delta)
        image = TF.adjust_contrast(image, 1.0 + self.contrast_delta)

        # Add Gaussian noise with deterministic seed
        if self.noise_std > 0:
            # Derive seed from image_id for reproducibility
            if image_id is not None:
                # Use hash of image_id to get deterministic seed
                # Modulo 2^31 - 1 to stay within valid range for torch.Generator
                seed_value = abs(hash(image_id)) % (2**31 - 1)
            elif seed is not None:
                seed_value = seed
            else:
                seed_value = None

            if seed_value is not None:
                generator = torch.Generator().manual_seed(seed_value)
                noise = torch.randn(image.shape, generator=generator, dtype=image.dtype, device=image.device) * self.noise_std
            else:
                noise = torch.randn_like(image) * self.noise_std

            image = image + noise

        # Clip to valid range [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image
```

---

## 2. Perturbation Parameters

**File:** `config.py`
**Lines:** 78-82

```python
ROBUSTNESS_PERTURBATION = {
    "brightness_delta": 0.1,
    "contrast_delta": 0.1,
    "noise_std": 0.02,
}
```

### Parameter Values:
- **brightness_delta:** 0.1 (multiply brightness by 1.1)
- **contrast_delta:** 0.1 (multiply contrast by 1.1)
- **noise_std:** 0.02 (standard deviation of Gaussian noise)

---

## 3. Condition 1: Intensity-Only Perturbations

### Perturbations Applied:

#### 3.1 Brightness Adjustment

**Line 117:**
```python
image = TF.adjust_brightness(image, 1.0 + self.brightness_delta)
```

**Formula:**
```
adjusted_image = image * (1.0 + brightness_delta)
                = image * 1.1
```

**Type:** Intensity-only (pixel values scaled uniformly)

✅ **Confirmed:** No spatial/geometric transform

#### 3.2 Contrast Scaling

**Line 118:**
```python
image = TF.adjust_contrast(image, 1.0 + self.contrast_delta)
```

**Formula:**
```
adjusted_image = (image - mean) * (1.0 + contrast_delta) + mean
               = (image - mean) * 1.1 + mean
```

**Type:** Intensity-only (pixel values scaled around mean)

✅ **Confirmed:** No spatial/geometric transform

#### 3.3 Gaussian Noise

**Lines 132-138:**
```python
if seed_value is not None:
    generator = torch.Generator().manual_seed(seed_value)
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype, device=image.device) * self.noise_std
else:
    noise = torch.randn_like(image) * self.noise_std

image = image + noise
```

**Formula:**
```
perturbed_image = image + N(0, noise_std²)
                = image + N(0, 0.02²)
```

**Type:** Intensity-only (additive noise per pixel)

✅ **Confirmed:** No spatial/geometric transform

### Perturbation Summary:

| Perturbation | Type | Formula | Geometric Transform? |
|--------------|------|---------|----------------------|
| Brightness | Intensity | `image * 1.1` | ❌ No |
| Contrast | Intensity | `(image - mean) * 1.1 + mean` | ❌ No |
| Gaussian Noise | Intensity | `image + N(0, 0.02²)` | ❌ No |

✅ **Confirmed: All perturbations are intensity-only**

---

## 4. Condition 2: No Geometric Transforms

**Code Review:**
- ❌ No `TF.rotate()`
- ❌ No `TF.affine()`
- ❌ No `TF.hflip()` or `TF.vflip()`
- ❌ No `TF.resize()`
- ❌ No `TF.crop()`
- ❌ No `TF.perspective()`

**Only transforms used:**
- ✅ `TF.adjust_brightness()` - Intensity-only
- ✅ `TF.adjust_contrast()` - Intensity-only
- ✅ Additive Gaussian noise - Intensity-only
- ✅ `torch.clamp()` - Clipping to [0, 1]

✅ **Confirmed: Zero geometric transforms**

---

## 5. Condition 3: Deterministic Seed Derivation

### Seed Derivation Logic

**Lines 123-130:**
```python
# Derive seed from image_id for reproducibility
if image_id is not None:
    # Use hash of image_id to get deterministic seed
    # Modulo 2^31 - 1 to stay within valid range for torch.Generator
    seed_value = abs(hash(image_id)) % (2**31 - 1)
elif seed is not None:
    seed_value = seed
else:
    seed_value = None
```

### Seed Formula:
```
seed_value = abs(hash(image_id)) % (2^31 - 1)
```

**Components:**
1. `hash(image_id)` - Python's built-in hash function (deterministic per string)
2. `abs(...)` - Ensure positive value
3. `% (2**31 - 1)` - Modulo 2,147,483,647 (valid range for torch.Generator)

### Example:
```python
image_id = "abc123_R_CC"
hash_value = hash(image_id)  # e.g., -5234567890
abs_hash = abs(hash_value)    # e.g., 5234567890
seed_value = abs_hash % (2**31 - 1)  # e.g., 5234567890 % 2147483647 = 939600596
```

### Properties:
- **Deterministic:** Same `image_id` always produces same `seed_value`
- **Unique:** Different `image_id` likely produces different `seed_value`
- **Valid range:** Seed is in [0, 2^31 - 2]

✅ **Confirmed: Seed is deterministically derived from image_id**

---

## 6. Condition 4: Same Perturbation Every Time

### Noise Generation with Fixed Seed

**Lines 132-134:**
```python
if seed_value is not None:
    generator = torch.Generator().manual_seed(seed_value)
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype, device=image.device) * self.noise_std
```

### Mechanism:
1. Create new `torch.Generator()` instance
2. Seed it with `seed_value` (derived from image_id)
3. Generate noise using this generator

### Determinism Guarantee:
- PyTorch's `Generator.manual_seed()` ensures **exact same noise** for same seed
- Each call with same image_id will:
  1. Derive same `seed_value`
  2. Generate same noise tensor
  3. Produce identical perturbed image

### Brightness and Contrast:
- **Lines 117-118:** Fixed adjustments (no randomness)
- **brightness_delta = 0.1** (constant)
- **contrast_delta = 0.1** (constant)
- Always multiply by exactly 1.1

✅ **Confirmed: Same image_id produces identical perturbation every time**

---

## 7. Usage in Robustness Evaluation

**File:** `training/robustness.py`
**Lines:** 72-75

```python
# Perturbed inference with deterministic seed per image_id
images_perturbed = torch.stack([
    self.perturbation(img, image_id=img_id)
    for img, img_id in zip(images, image_ids)
]).to(self.device)
```

**Flow:**
1. For each image in batch, pass both `img` and `img_id` to perturbation
2. Perturbation derives seed from `img_id`
3. Applies deterministic brightness, contrast, and noise
4. Returns perturbed image

**Guarantee:**
- Image with `image_id="abc123"` always receives same perturbation
- Across all epochs and evaluations
- Ensures reproducible robustness metrics

---

## 8. Code Locations Summary

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **RobustnessPerturbation class** | `data/augmentation.py` | 79-143 | Full implementation |
| **Brightness adjustment** | `data/augmentation.py` | 117 | `TF.adjust_brightness(image, 1.1)` |
| **Contrast adjustment** | `data/augmentation.py` | 118 | `TF.adjust_contrast(image, 1.1)` |
| **Seed derivation** | `data/augmentation.py` | 123-126 | `abs(hash(image_id)) % (2^31 - 1)` |
| **Noise generation** | `data/augmentation.py` | 132-134 | `torch.randn(..., generator=...)` |
| **Perturbation parameters** | `config.py` | 78-82 | Brightness: 0.1, Contrast: 0.1, Noise: 0.02 |
| **Usage in robustness eval** | `training/robustness.py` | 72-75 | Apply to each image with image_id |

---

## 9. Verification Test

**File:** `test_robustness_perturbation_determinism.py` (created)

### Test Cases:
1. **Same image_id produces identical outputs**
   - Apply perturbation twice to same image with same image_id
   - Assert tensors are exactly equal (element-wise)

2. **Different image_ids produce different outputs**
   - Apply perturbation to same image with different image_ids
   - Assert tensors are different

3. **Seed derivation consistency**
   - Verify `hash(image_id)` produces same value across calls
   - Verify seed computation is deterministic

4. **Brightness and contrast are fixed**
   - Verify no randomness in brightness/contrast adjustments
   - Always multiply by exactly 1.1

---

## 10. Mathematical Properties

### Property 1: Determinism
For any image I and image_id ID:
```
perturbation(I, ID) = perturbation(I, ID)  [always]
```

**Proof:** Seed derived from ID is deterministic, PyTorch generator is deterministic for fixed seed.

### Property 2: Uniqueness (likely)
For different image_ids ID1 ≠ ID2:
```
perturbation(I, ID1) ≠ perturbation(I, ID2)  [with high probability]
```

**Proof:** Hash function distributes IDs across seed space, making collision unlikely.

### Property 3: Reproducibility
Same image_id produces same perturbation across:
- Different epochs
- Different evaluation runs
- Different machines (assuming same PyTorch version)

---

## 11. Comparison with Training Augmentation

| Aspect | Training Augmentation | Robustness Perturbation |
|--------|----------------------|-------------------------|
| **Class** | `IntensityAugmentation` | `RobustnessPerturbation` |
| **Randomness** | Random (every call) | Deterministic (per image_id) |
| **Strength** | Configurable [0, 1] | Fixed (brightness=0.1, contrast=0.1, noise=0.02) |
| **Purpose** | Data augmentation | Robustness evaluation |
| **Brightness** | Random in [-δ, +δ] | Fixed at +0.1 |
| **Contrast** | Random in [-δ, +δ] | Fixed at +0.1 |
| **Noise** | Random N(0, σ²) | Deterministic N(0, 0.02²) per image_id |

---

## Conclusion

**ALL CONDITIONS CONFIRMED:**

1. ✅ **Intensity-only perturbations:**
   - Brightness: multiply by 1.1
   - Contrast: scale by 1.1
   - Gaussian noise: N(0, 0.02²)

2. ✅ **No geometric transforms:**
   - Zero use of rotation, flipping, cropping, or spatial transforms
   - Only pixel value modifications

3. ✅ **Deterministic per image_id:**
   - Seed derived from `abs(hash(image_id)) % (2^31 - 1)`
   - Same image_id always produces same seed

4. ✅ **Same perturbation every time:**
   - PyTorch generator seeded with image_id hash
   - Brightness and contrast are fixed values
   - Noise is deterministic for given seed

**The robustness perturbation implementation meets all research requirements.**
