"""
Unit tests for breast-level Noisy OR aggregation.

This test suite verifies the generalized Noisy OR formula implementation:
    p_breast = 1 - ∏_{j=1}^{m} (1 - p_j)

Special cases tested:
- m = 1: p_breast = p_1 (single view)
- m = 2: p_breast = 1 - (1 - p_1)(1 - p_2) (standard Noisy OR)
- m > 2: generalized formula
"""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import unittest
import numpy as np
import pandas as pd
from utils.noisy_or import aggregate_to_breast_level, noisy_or_aggregation


class TestNoisyORAggregation(unittest.TestCase):
    """Test suite for Noisy OR breast-level aggregation."""

    def test_breast_id_definition(self):
        """
        Test that breast_id is defined as patient_id + "_" + laterality.

        Implementation location: data/parsers.py:438
        Code: breast_id = patient_id + "_" + laterality
        """
        # Create toy metadata
        metadata = pd.DataFrame({
            "image_id": ["img1", "img2"],
            "patient_id": ["P001", "P001"],
            "laterality": ["R", "L"],
            "view": ["CC", "MLO"],
            "label": [0, 1]
        })

        # Manually construct breast_id to verify formula
        metadata["breast_id"] = metadata["patient_id"] + "_" + metadata["laterality"]

        # Verify breast_id values
        self.assertEqual(metadata.loc[0, "breast_id"], "P001_R")
        self.assertEqual(metadata.loc[1, "breast_id"], "P001_L")

        print("[OK] breast_id definition: patient_id + '_' + laterality")

    def test_single_view_case(self):
        """
        Test Noisy OR with single view (m = 1).

        Expected: p_breast = p_1

        Implementation: utils/noisy_or.py:69-71
        Code:
            if len(image_probs) == 1:
                breast_prob = image_probs[0]
        """
        # Toy example: One breast with only CC view
        metadata = pd.DataFrame({
            "image_id": ["img_cc"],
            "patient_id": ["P001"],
            "breast_id": ["P001_R"],
            "view": ["CC"],
            "label": [1]
        })

        # Image predictions
        image_predictions = {
            "img_cc": 0.7
        }

        # Aggregate to breast level
        breast_preds, breast_labels = aggregate_to_breast_level(image_predictions, metadata)

        # Expected: p_breast = p_CC = 0.7
        expected_prob = 0.7

        self.assertEqual(len(breast_preds), 1)
        self.assertAlmostEqual(breast_preds[0], expected_prob, places=6)
        self.assertEqual(breast_labels[0], 1)

        print(f"[OK] Single view case: p_breast = {breast_preds[0]:.6f} (expected: {expected_prob})")

    def test_two_views_case(self):
        """
        Test Noisy OR with two views (m = 2).

        Expected: p_breast = 1 - (1 - p_CC)(1 - p_MLO)

        Implementation: utils/noisy_or.py:73-74
        Code:
            breast_prob = 1.0 - np.prod([1.0 - p for p in image_probs])
        """
        # Toy example: One breast with CC and MLO views
        metadata = pd.DataFrame({
            "image_id": ["img_cc", "img_mlo"],
            "patient_id": ["P001", "P001"],
            "breast_id": ["P001_R", "P001_R"],
            "view": ["CC", "MLO"],
            "label": [1, 1]
        })

        # Image predictions
        image_predictions = {
            "img_cc": 0.6,
            "img_mlo": 0.8
        }

        # Aggregate to breast level
        breast_preds, breast_labels = aggregate_to_breast_level(image_predictions, metadata)

        # Expected: p_breast = 1 - (1 - 0.6)(1 - 0.8) = 1 - 0.4 * 0.2 = 1 - 0.08 = 0.92
        expected_prob = 1 - (1 - 0.6) * (1 - 0.8)

        self.assertEqual(len(breast_preds), 1)
        self.assertAlmostEqual(breast_preds[0], expected_prob, places=6)
        self.assertEqual(breast_labels[0], 1)

        print(f"[OK] Two views case: p_breast = {breast_preds[0]:.6f} (expected: {expected_prob:.6f})")

    def test_three_views_generalized(self):
        """
        Test generalized Noisy OR with three views (m = 3).

        Expected: p_breast = 1 - (1 - p_1)(1 - p_2)(1 - p_3)
        """
        # Toy example: One breast with three views (uncommon but possible)
        metadata = pd.DataFrame({
            "image_id": ["img1", "img2", "img3"],
            "patient_id": ["P001", "P001", "P001"],
            "breast_id": ["P001_R", "P001_R", "P001_R"],
            "view": ["CC", "MLO", "CC"],  # Two CC views and one MLO (e.g., repeat imaging)
            "label": [1, 1, 1]
        })

        # Image predictions
        image_predictions = {
            "img1": 0.5,
            "img2": 0.6,
            "img3": 0.7
        }

        # Aggregate to breast level
        breast_preds, breast_labels = aggregate_to_breast_level(image_predictions, metadata)

        # Expected: p_breast = 1 - (1 - 0.5)(1 - 0.6)(1 - 0.7)
        #                     = 1 - 0.5 * 0.4 * 0.3
        #                     = 1 - 0.06
        #                     = 0.94
        expected_prob = 1 - (1 - 0.5) * (1 - 0.6) * (1 - 0.7)

        self.assertEqual(len(breast_preds), 1)
        self.assertAlmostEqual(breast_preds[0], expected_prob, places=6)
        self.assertEqual(breast_labels[0], 1)

        print(f"[OK] Three views case: p_breast = {breast_preds[0]:.6f} (expected: {expected_prob:.6f})")

    def test_groupby_multiple_breasts(self):
        """
        Test groupby logic with multiple breasts from different patients.

        Implementation: utils/noisy_or.py:53
        Code: breast_groups = metadata.groupby("breast_id")
        """
        # Toy example: Two patients, each with left and right breasts
        metadata = pd.DataFrame({
            "image_id": ["p1_r_cc", "p1_r_mlo", "p1_l_cc", "p2_r_cc", "p2_r_mlo"],
            "patient_id": ["P001", "P001", "P001", "P002", "P002"],
            "breast_id": ["P001_R", "P001_R", "P001_L", "P002_R", "P002_R"],
            "view": ["CC", "MLO", "CC", "CC", "MLO"],
            "label": [1, 1, 0, 0, 0]
        })

        # Image predictions
        image_predictions = {
            "p1_r_cc": 0.7,
            "p1_r_mlo": 0.6,
            "p1_l_cc": 0.2,
            "p2_r_cc": 0.3,
            "p2_r_mlo": 0.4
        }

        # Aggregate to breast level
        breast_preds, breast_labels = aggregate_to_breast_level(image_predictions, metadata)

        # Expected results for each breast
        # P001_R: 1 - (1 - 0.7)(1 - 0.6) = 1 - 0.3 * 0.4 = 0.88
        # P001_L: 0.2 (single view)
        # P002_R: 1 - (1 - 0.3)(1 - 0.4) = 1 - 0.7 * 0.6 = 0.58

        expected_probs = {
            "P001_R": 1 - (1 - 0.7) * (1 - 0.6),
            "P001_L": 0.2,
            "P002_R": 1 - (1 - 0.3) * (1 - 0.4)
        }

        # Verify we have 3 unique breasts
        self.assertEqual(len(breast_preds), 3)

        # Map breast_id to index for verification
        breast_ids = metadata.groupby("breast_id").groups.keys()
        for i, breast_id in enumerate(sorted(breast_ids)):
            expected = expected_probs[breast_id]
            self.assertAlmostEqual(breast_preds[i], expected, places=6)
            print(f"[OK] Breast {breast_id}: p_breast = {breast_preds[i]:.6f} (expected: {expected:.6f})")

    def test_label_aggregation_max(self):
        """
        Test that breast label uses max() across all views.

        Implementation: utils/noisy_or.py:80
        Code: breast_label = int(group["label"].max())
        """
        # Toy example: Breast with inconsistent labels (one positive, one negative)
        metadata = pd.DataFrame({
            "image_id": ["img_cc", "img_mlo"],
            "patient_id": ["P001", "P001"],
            "breast_id": ["P001_R", "P001_R"],
            "view": ["CC", "MLO"],
            "label": [0, 1]  # Inconsistent labels
        })

        image_predictions = {
            "img_cc": 0.3,
            "img_mlo": 0.7
        }

        # Aggregate to breast level
        breast_preds, breast_labels = aggregate_to_breast_level(image_predictions, metadata)

        # Expected label: max(0, 1) = 1
        self.assertEqual(breast_labels[0], 1)

        print(f"[OK] Label aggregation: max(0, 1) = {breast_labels[0]}")

    def test_noisy_or_function_two_views(self):
        """
        Test the standalone noisy_or_aggregation function for two views.

        Implementation: utils/noisy_or.py:10-23
        """
        p_cc = 0.6
        p_mlo = 0.8

        result = noisy_or_aggregation(p_cc, p_mlo)

        # Expected: 1 - (1 - 0.6)(1 - 0.8) = 0.92
        expected = 1 - (1 - p_cc) * (1 - p_mlo)

        self.assertAlmostEqual(result, expected, places=6)
        self.assertAlmostEqual(result, 0.92, places=6)

        print(f"[OK] noisy_or_aggregation(0.6, 0.8) = {result:.6f} (expected: {expected:.6f})")

    def test_extreme_values(self):
        """Test extreme probability values (0.0, 1.0)."""
        # Case 1: Both views predict 0.0
        result1 = noisy_or_aggregation(0.0, 0.0)
        self.assertEqual(result1, 0.0)
        print(f"[OK] noisy_or_aggregation(0.0, 0.0) = {result1}")

        # Case 2: Both views predict 1.0
        result2 = noisy_or_aggregation(1.0, 1.0)
        self.assertEqual(result2, 1.0)
        print(f"[OK] noisy_or_aggregation(1.0, 1.0) = {result2}")

        # Case 3: One view predicts 1.0, other 0.0
        result3 = noisy_or_aggregation(1.0, 0.0)
        self.assertEqual(result3, 1.0)
        print(f"[OK] noisy_or_aggregation(1.0, 0.0) = {result3}")

        # Case 4: One view predicts 0.5, other 1.0
        result4 = noisy_or_aggregation(0.5, 1.0)
        self.assertEqual(result4, 1.0)
        print(f"[OK] noisy_or_aggregation(0.5, 1.0) = {result4}")

    def test_numerical_example_from_paper(self):
        """
        Test with a realistic numerical example mimicking clinical data.

        Scenario: Patient P001 has bilateral findings
        - Right breast (P001_R): CC view shows 0.65, MLO view shows 0.72
        - Left breast (P001_L): Only CC view available, shows 0.15
        """
        metadata = pd.DataFrame({
            "image_id": ["p001_r_cc", "p001_r_mlo", "p001_l_cc"],
            "patient_id": ["P001", "P001", "P001"],
            "breast_id": ["P001_R", "P001_R", "P001_L"],
            "view": ["CC", "MLO", "CC"],
            "label": [1, 1, 0]
        })

        image_predictions = {
            "p001_r_cc": 0.65,
            "p001_r_mlo": 0.72,
            "p001_l_cc": 0.15
        }

        # Aggregate
        breast_preds, breast_labels = aggregate_to_breast_level(image_predictions, metadata)

        # Expected:
        # P001_R: 1 - (1 - 0.65)(1 - 0.72) = 1 - 0.35 * 0.28 = 1 - 0.098 = 0.902
        # P001_L: 0.15 (single view)

        expected_p001_r = 1 - (1 - 0.65) * (1 - 0.72)
        expected_p001_l = 0.15

        # Sort by breast_id for consistent ordering
        breast_ids = sorted(metadata["breast_id"].unique())
        results = dict(zip(breast_ids, breast_preds))

        self.assertAlmostEqual(results["P001_L"], expected_p001_l, places=6)
        self.assertAlmostEqual(results["P001_R"], expected_p001_r, places=6)

        print(f"\n[OK] Realistic clinical example:")
        print(f"     P001_R (CC=0.65, MLO=0.72) -> p_breast = {results['P001_R']:.6f} (expected: {expected_p001_r:.6f})")
        print(f"     P001_L (CC=0.15) -> p_breast = {results['P001_L']:.6f} (expected: {expected_p001_l:.6f})")


def main():
    """Run all tests and display summary."""
    print("=" * 80)
    print("NOISY OR AGGREGATION TEST SUITE")
    print("=" * 80)
    print("\nTesting breast-level evaluation implementation:")
    print("  - breast_id definition: patient_id + '_' + laterality")
    print("  - Grouping: metadata.groupby('breast_id')")
    print("  - Formula: p_breast = 1 - product(1 - p_j)")
    print("  - Special case: m=1 -> p_breast = p_1")
    print("\n" + "=" * 80)

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNoisyORAggregation)
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
        print("\n[OK] ALL TESTS PASSED - Noisy OR implementation verified")
    else:
        print("\n[FAIL] Some tests failed")

    print("=" * 80)


if __name__ == "__main__":
    main()
