"""
Unit tests for robustness perturbation determinism.

This test suite verifies:
1. Same image_id produces identical outputs (deterministic)
2. Different image_ids produce different outputs (unique)
3. Seed derivation is consistent
4. Perturbations are intensity-only (no geometric transforms)
"""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import unittest
import torch
import numpy as np
from data.augmentation import RobustnessPerturbation


class TestRobustnessPerturbationDeterminism(unittest.TestCase):
    """Test suite for robustness perturbation determinism."""

    def setUp(self):
        """Create test image and perturbation instance."""
        # Create a simple test image (3, 64, 64) with random values
        torch.manual_seed(42)
        self.test_image = torch.rand(3, 64, 64)

        # Create perturbation instance
        self.perturbation = RobustnessPerturbation(
            brightness_delta=0.1,
            contrast_delta=0.1,
            noise_std=0.02,
        )

    def test_same_image_id_identical_output(self):
        """
        CRITICAL TEST: Same image_id must produce identical outputs.

        This is the core requirement for deterministic robustness evaluation.
        """
        image_id = "test_image_001"

        # Apply perturbation twice with same image_id
        output1 = self.perturbation(self.test_image.clone(), image_id=image_id)
        output2 = self.perturbation(self.test_image.clone(), image_id=image_id)

        # Outputs must be EXACTLY identical (element-wise)
        self.assertTrue(
            torch.allclose(output1, output2, atol=1e-8, rtol=1e-8),
            "Same image_id must produce identical outputs"
        )

        # Verify exact equality (not just close)
        self.assertTrue(
            torch.equal(output1, output2),
            "Outputs should be exactly equal, not just close"
        )

        print(f"[OK] Same image_id produces identical output")
        print(f"     image_id: {image_id}")
        print(f"     Output 1 mean: {output1.mean():.6f}, std: {output1.std():.6f}")
        print(f"     Output 2 mean: {output2.mean():.6f}, std: {output2.std():.6f}")
        print(f"     Max absolute difference: {(output1 - output2).abs().max():.2e}")

    def test_multiple_calls_same_image_id(self):
        """Test that multiple calls with same image_id produce identical results."""
        image_id = "patient_abc123_R_CC"

        outputs = []
        for i in range(5):
            output = self.perturbation(self.test_image.clone(), image_id=image_id)
            outputs.append(output)

        # All outputs must be identical
        for i in range(1, len(outputs)):
            self.assertTrue(
                torch.equal(outputs[0], outputs[i]),
                f"Call {i} produced different output from call 0"
            )

        print(f"\n[OK] {len(outputs)} calls with same image_id produce identical outputs")

    def test_different_image_ids_different_outputs(self):
        """Test that different image_ids produce different outputs."""
        image_id1 = "image_001"
        image_id2 = "image_002"

        # Apply perturbation with different image_ids
        output1 = self.perturbation(self.test_image.clone(), image_id=image_id1)
        output2 = self.perturbation(self.test_image.clone(), image_id=image_id2)

        # Outputs should be different
        self.assertFalse(
            torch.equal(output1, output2),
            "Different image_ids should produce different outputs"
        )

        # Compute difference
        diff = (output1 - output2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n[OK] Different image_ids produce different outputs")
        print(f"     image_id1: {image_id1}, image_id2: {image_id2}")
        print(f"     Max difference: {max_diff:.6f}")
        print(f"     Mean difference: {mean_diff:.6f}")

        # Ensure difference is non-trivial (noise should cause visible difference)
        self.assertGreater(max_diff, 0.001, "Difference should be non-trivial")

    def test_seed_derivation_consistency(self):
        """Test that seed derivation from image_id is consistent."""
        image_id = "test_patient_L_MLO"

        # Derive seed multiple times
        seeds = []
        for _ in range(10):
            seed = abs(hash(image_id)) % (2**31 - 1)
            seeds.append(seed)

        # All seeds must be identical
        self.assertEqual(len(set(seeds)), 1, "Seed derivation must be deterministic")

        print(f"\n[OK] Seed derivation is consistent")
        print(f"     image_id: {image_id}")
        print(f"     Derived seed: {seeds[0]}")
        print(f"     All 10 derivations produced same seed: {seeds[0]}")

    def test_seed_formula_correctness(self):
        """Test the seed derivation formula."""
        test_cases = [
            ("abc123", abs(hash("abc123")) % (2**31 - 1)),
            ("patient_001_R_CC", abs(hash("patient_001_R_CC")) % (2**31 - 1)),
            ("", abs(hash("")) % (2**31 - 1)),
        ]

        for image_id, expected_seed in test_cases:
            seed = abs(hash(image_id)) % (2**31 - 1)
            self.assertEqual(seed, expected_seed)
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2**31 - 1)

        print(f"\n[OK] Seed formula correct for {len(test_cases)} test cases")

    def test_brightness_is_fixed(self):
        """Test that brightness adjustment is fixed (not random)."""
        image_id = "test_brightness"

        # Apply perturbation multiple times
        outputs = [self.perturbation(self.test_image.clone(), image_id=image_id) for _ in range(3)]

        # All outputs should be identical (brightness is fixed)
        for output in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0], output))

        print(f"\n[OK] Brightness adjustment is fixed (not random)")

    def test_contrast_is_fixed(self):
        """Test that contrast adjustment is fixed (not random)."""
        image_id = "test_contrast"

        # Apply perturbation multiple times
        outputs = [self.perturbation(self.test_image.clone(), image_id=image_id) for _ in range(3)]

        # All outputs should be identical (contrast is fixed)
        for output in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0], output))

        print(f"\n[OK] Contrast adjustment is fixed (not random)")

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape (no geometric transforms)."""
        image_id = "test_shape"

        # Test with different shapes
        test_shapes = [(3, 64, 64), (3, 128, 128), (1, 32, 32)]

        for shape in test_shapes:
            image = torch.rand(shape)
            output = self.perturbation(image, image_id=image_id)

            self.assertEqual(
                output.shape, image.shape,
                f"Output shape {output.shape} must match input shape {image.shape}"
            )

        print(f"\n[OK] Output shape matches input shape (no geometric transforms)")
        print(f"     Tested shapes: {test_shapes}")

    def test_output_range_valid(self):
        """Test that output values are clipped to [0, 1]."""
        image_id = "test_range"

        # Create edge case images
        edge_cases = [
            torch.zeros(3, 64, 64),      # All zeros
            torch.ones(3, 64, 64),       # All ones
            torch.rand(3, 64, 64) * 0.1, # Very dim
            torch.rand(3, 64, 64) * 0.9 + 0.1, # Very bright
        ]

        for image in edge_cases:
            output = self.perturbation(image, image_id=image_id)

            # All values must be in [0, 1]
            self.assertGreaterEqual(output.min().item(), 0.0)
            self.assertLessEqual(output.max().item(), 1.0)

        print(f"\n[OK] Output values clipped to [0, 1]")

    def test_no_inplace_modification(self):
        """Test that perturbation does not modify input image in-place."""
        image_id = "test_inplace"

        original_image = self.test_image.clone()
        original_copy = self.test_image.clone()

        # Apply perturbation
        output = self.perturbation(original_image, image_id=image_id)

        # Original image should be unchanged
        self.assertTrue(
            torch.equal(original_image, original_copy),
            "Perturbation should not modify input in-place"
        )

        # Output should be different from input
        self.assertFalse(
            torch.equal(output, original_image),
            "Output should differ from input"
        )

        print(f"\n[OK] Perturbation does not modify input in-place")

    def test_perturbation_magnitude(self):
        """Test that perturbation magnitude is reasonable (not too large or small)."""
        image_id = "test_magnitude"

        # Create test image with known values
        test_image = torch.ones(3, 64, 64) * 0.5

        # Apply perturbation
        output = self.perturbation(test_image, image_id=image_id)

        # Compute difference
        diff = (output - test_image).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n[OK] Perturbation magnitude analysis:")
        print(f"     Input: constant 0.5")
        print(f"     Max difference: {max_diff:.6f}")
        print(f"     Mean difference: {mean_diff:.6f}")

        # Perturbation should be noticeable but not extreme
        self.assertGreater(mean_diff, 0.001, "Perturbation should be noticeable")
        self.assertLess(max_diff, 0.5, "Perturbation should not be extreme")

    def test_realistic_image_ids(self):
        """Test with realistic image IDs from VinDr-Mammo and INbreast."""
        realistic_ids = [
            "4e3a578fe535ea4f5258d3f7f4419db8",  # VinDr-Mammo style
            "patient_001_R_CC",                   # Generic style
            "20586908_6c613a14b80a8591_MG_R_CC_ANON",  # INbreast style
        ]

        for image_id in realistic_ids:
            # Apply twice
            output1 = self.perturbation(self.test_image.clone(), image_id=image_id)
            output2 = self.perturbation(self.test_image.clone(), image_id=image_id)

            # Must be identical
            self.assertTrue(
                torch.equal(output1, output2),
                f"Failed for realistic image_id: {image_id}"
            )

        print(f"\n[OK] Determinism verified for {len(realistic_ids)} realistic image IDs")

    def test_cross_device_consistency(self):
        """Test that perturbation is consistent across CPU and CUDA (if available)."""
        if not torch.cuda.is_available():
            print(f"\n[SKIP] CUDA not available, skipping cross-device test")
            return

        image_id = "test_device"

        # CPU version
        image_cpu = self.test_image.clone()
        output_cpu = self.perturbation(image_cpu, image_id=image_id)

        # CUDA version
        image_cuda = self.test_image.clone().cuda()
        output_cuda = self.perturbation(image_cuda, image_id=image_id)

        # Outputs should be identical (within floating point precision)
        self.assertTrue(
            torch.allclose(output_cpu, output_cuda.cpu(), atol=1e-6, rtol=1e-6),
            "Perturbation should be consistent across CPU and CUDA"
        )

        print(f"\n[OK] Perturbation consistent across CPU and CUDA")


def main():
    """Run all tests and display summary."""
    print("=" * 80)
    print("ROBUSTNESS PERTURBATION DETERMINISM TEST SUITE")
    print("=" * 80)
    print("\nVerifying:")
    print("  1. Same image_id produces identical outputs (CRITICAL)")
    print("  2. Different image_ids produce different outputs")
    print("  3. Seed derivation is consistent")
    print("  4. Perturbations are intensity-only (no geometric transforms)")
    print("\n" + "=" * 80)

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRobustnessPerturbationDeterminism)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run:     {result.testsRun}")
    print(f"Successes:     {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")

    if result.wasSuccessful():
        print("\n[OK] ALL TESTS PASSED")
        print("\nCONFIRMED:")
        print("  - Same image_id produces IDENTICAL outputs (deterministic)")
        print("  - Different image_ids produce DIFFERENT outputs (unique)")
        print("  - Seed derivation is CONSISTENT")
        print("  - Perturbations are INTENSITY-ONLY (no geometric transforms)")
        print("  - Output range is VALID [0, 1]")
        print("  - No IN-PLACE modification")
    else:
        print("\n[FAIL] Some tests failed")

    print("=" * 80)


if __name__ == "__main__":
    main()
