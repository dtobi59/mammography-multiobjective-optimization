"""
Unit tests for Brier score and derived metrics.

This test suite verifies:
1. Brier_null = p * (1 - p) where p = prevalence
2. Scaled Brier = 1 - (Brier / Brier_null)
3. Edge cases (all zeros, all ones, balanced, imbalanced)
4. Numerical examples with known outcomes
"""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import unittest
import numpy as np
from training.metrics import compute_brier_null, compute_scaled_brier, compute_metrics
from sklearn.metrics import brier_score_loss


class TestBrierMetrics(unittest.TestCase):
    """Test suite for Brier score and derived metrics."""

    def test_prevalence_computation(self):
        """Test that prevalence is computed correctly."""
        # Test case 1: 60% positive
        labels1 = np.array([0, 0, 1, 1, 1])
        prevalence1 = labels1.mean()
        self.assertAlmostEqual(prevalence1, 0.6, places=6)

        # Test case 2: 25% positive
        labels2 = np.array([0, 0, 0, 1])
        prevalence2 = labels2.mean()
        self.assertAlmostEqual(prevalence2, 0.25, places=6)

        # Test case 3: 50% positive (balanced)
        labels3 = np.array([0, 0, 1, 1])
        prevalence3 = labels3.mean()
        self.assertAlmostEqual(prevalence3, 0.5, places=6)

        print(f"\n[OK] Prevalence computation verified")
        print(f"     60% positive: p = {prevalence1:.2f}")
        print(f"     25% positive: p = {prevalence2:.2f}")
        print(f"     50% positive: p = {prevalence3:.2f}")

    def test_brier_null_formula(self):
        """Test that Brier_null = p * (1 - p)."""
        # Test case 1: p = 0.6
        labels1 = np.array([0, 0, 1, 1, 1])
        prevalence1 = labels1.mean()  # 0.6
        brier_null1 = compute_brier_null(labels1)
        expected1 = prevalence1 * (1 - prevalence1)  # 0.6 * 0.4 = 0.24

        self.assertAlmostEqual(brier_null1, expected1, places=6)
        self.assertAlmostEqual(brier_null1, 0.24, places=6)

        # Test case 2: p = 0.25
        labels2 = np.array([0, 0, 0, 1])
        prevalence2 = labels2.mean()  # 0.25
        brier_null2 = compute_brier_null(labels2)
        expected2 = prevalence2 * (1 - prevalence2)  # 0.25 * 0.75 = 0.1875

        self.assertAlmostEqual(brier_null2, expected2, places=6)
        self.assertAlmostEqual(brier_null2, 0.1875, places=6)

        # Test case 3: p = 0.5 (maximum Brier_null)
        labels3 = np.array([0, 0, 1, 1])
        prevalence3 = labels3.mean()  # 0.5
        brier_null3 = compute_brier_null(labels3)
        expected3 = prevalence3 * (1 - prevalence3)  # 0.5 * 0.5 = 0.25

        self.assertAlmostEqual(brier_null3, expected3, places=6)
        self.assertAlmostEqual(brier_null3, 0.25, places=6)

        print(f"\n[OK] Brier_null formula verified: p * (1 - p)")
        print(f"     p = 0.60: Brier_null = {brier_null1:.4f} (expected: 0.2400)")
        print(f"     p = 0.25: Brier_null = {brier_null2:.4f} (expected: 0.1875)")
        print(f"     p = 0.50: Brier_null = {brier_null3:.4f} (expected: 0.2500, maximum)")

    def test_brier_null_direct_calculation(self):
        """Verify Brier_null by direct calculation of mean((p - y_i)^2)."""
        labels = np.array([0, 0, 1, 1, 1])
        prevalence = labels.mean()  # 0.6

        # Direct calculation
        brier_null_direct = np.mean((prevalence - labels) ** 2)
        # Expected: mean([0.36, 0.36, 0.16, 0.16, 0.16]) = 1.2 / 5 = 0.24

        # Formula calculation
        brier_null_formula = compute_brier_null(labels)

        self.assertAlmostEqual(brier_null_direct, 0.24, places=6)
        self.assertAlmostEqual(brier_null_formula, brier_null_direct, places=6)

        print(f"\n[OK] Brier_null verified by direct calculation")
        print(f"     Direct: mean((p - y)^2) = {brier_null_direct:.4f}")
        print(f"     Formula: p * (1 - p) = {brier_null_formula:.4f}")
        print(f"     Match: {np.isclose(brier_null_direct, brier_null_formula)}")

    def test_scaled_brier_formula(self):
        """Test that Scaled Brier = 1 - (Brier / Brier_null)."""
        labels = np.array([0, 0, 1, 1, 1])

        # Perfect predictions
        predictions_perfect = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        brier_perfect = brier_score_loss(labels, predictions_perfect)
        scaled_brier_perfect = compute_scaled_brier(brier_perfect, labels)

        # Brier should be 0, Scaled Brier should be 1
        self.assertAlmostEqual(brier_perfect, 0.0, places=6)
        self.assertAlmostEqual(scaled_brier_perfect, 1.0, places=6)

        # Predicting prevalence (null model)
        prevalence = labels.mean()  # 0.6
        predictions_null = np.full_like(labels, prevalence, dtype=float)
        brier_null = brier_score_loss(labels, predictions_null)
        scaled_brier_null = compute_scaled_brier(brier_null, labels)

        # Scaled Brier should be 0 (no better than baseline)
        self.assertAlmostEqual(scaled_brier_null, 0.0, places=6)

        # Good predictions (better than null)
        predictions_good = np.array([0.1, 0.1, 0.9, 0.9, 0.9])
        brier_good = brier_score_loss(labels, predictions_good)
        brier_null_value = compute_brier_null(labels)
        scaled_brier_good = compute_scaled_brier(brier_good, labels)
        expected_scaled_good = 1.0 - (brier_good / brier_null_value)

        self.assertAlmostEqual(scaled_brier_good, expected_scaled_good, places=6)
        self.assertGreater(scaled_brier_good, 0.0)  # Better than null

        print(f"\n[OK] Scaled Brier formula verified: 1 - (Brier / Brier_null)")
        print(f"     Perfect predictions: Scaled Brier = {scaled_brier_perfect:.4f} (expected: 1.0)")
        print(f"     Null predictions: Scaled Brier = {scaled_brier_null:.4f} (expected: 0.0)")
        print(f"     Good predictions: Scaled Brier = {scaled_brier_good:.4f} (expected: > 0.0)")

    def test_scaled_brier_range(self):
        """Test Scaled Brier range properties."""
        labels = np.array([0, 0, 1, 1, 1])

        # Perfect: Scaled Brier = 1
        predictions_perfect = labels.astype(float)
        scaled_perfect = compute_scaled_brier(brier_score_loss(labels, predictions_perfect), labels)
        self.assertAlmostEqual(scaled_perfect, 1.0, places=2)

        # Null: Scaled Brier = 0
        prevalence = labels.mean()
        predictions_null = np.full_like(labels, prevalence, dtype=float)
        scaled_null = compute_scaled_brier(brier_score_loss(labels, predictions_null), labels)
        self.assertAlmostEqual(scaled_null, 0.0, places=2)

        # Bad (reversed): Scaled Brier < 0
        predictions_bad = 1.0 - labels.astype(float)  # Predict opposite
        scaled_bad = compute_scaled_brier(brier_score_loss(labels, predictions_bad), labels)
        self.assertLess(scaled_bad, 0.0)

        print(f"\n[OK] Scaled Brier range properties verified")
        print(f"     Perfect model: {scaled_perfect:.4f} (expected: ~1.0)")
        print(f"     Null model: {scaled_null:.4f} (expected: 0.0)")
        print(f"     Bad model: {scaled_bad:.4f} (expected: < 0.0)")

    def test_compute_metrics_returns_all(self):
        """Test that compute_metrics() returns all 5 metrics."""
        labels = np.array([0, 0, 1, 1, 1])
        predictions = np.array([0.2, 0.3, 0.7, 0.8, 0.9])

        metrics = compute_metrics(predictions, labels)

        # Check all keys exist
        required_keys = ["pr_auc", "auroc", "brier", "brier_null", "scaled_brier"]
        for key in required_keys:
            self.assertIn(key, metrics)

        # Check values are reasonable
        self.assertGreater(metrics["pr_auc"], 0.0)
        self.assertLessEqual(metrics["pr_auc"], 1.0)
        self.assertGreaterEqual(metrics["auroc"], 0.5)
        self.assertLess(metrics["brier"], 1.0)
        self.assertGreater(metrics["brier_null"], 0.0)
        self.assertGreater(metrics["scaled_brier"], 0.0)

        print(f"\n[OK] compute_metrics() returns all 5 metrics")
        print(f"     PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"     AUROC: {metrics['auroc']:.4f}")
        print(f"     Brier: {metrics['brier']:.4f}")
        print(f"     Brier_null: {metrics['brier_null']:.4f}")
        print(f"     Scaled Brier: {metrics['scaled_brier']:.4f}")

    def test_edge_case_all_zeros(self):
        """Test edge case where all labels are 0."""
        labels = np.array([0, 0, 0, 0])
        predictions = np.array([0.1, 0.2, 0.3, 0.4])

        brier_null = compute_brier_null(labels)
        # p = 0, Brier_null = 0 * (1 - 0) = 0
        self.assertAlmostEqual(brier_null, 0.0, places=6)

        # Scaled Brier should handle division by zero
        brier = brier_score_loss(labels, predictions)
        scaled_brier = compute_scaled_brier(brier, labels)
        self.assertAlmostEqual(scaled_brier, 0.0, places=6)

        print(f"\n[OK] Edge case: all labels are 0")
        print(f"     Brier_null = {brier_null:.4f} (expected: 0.0)")
        print(f"     Scaled Brier = {scaled_brier:.4f} (expected: 0.0)")

    def test_edge_case_all_ones(self):
        """Test edge case where all labels are 1."""
        labels = np.array([1, 1, 1, 1])
        predictions = np.array([0.6, 0.7, 0.8, 0.9])

        brier_null = compute_brier_null(labels)
        # p = 1, Brier_null = 1 * (1 - 1) = 0
        self.assertAlmostEqual(brier_null, 0.0, places=6)

        # Scaled Brier should handle division by zero
        brier = brier_score_loss(labels, predictions)
        scaled_brier = compute_scaled_brier(brier, labels)
        self.assertAlmostEqual(scaled_brier, 0.0, places=6)

        print(f"\n[OK] Edge case: all labels are 1")
        print(f"     Brier_null = {brier_null:.4f} (expected: 0.0)")
        print(f"     Scaled Brier = {scaled_brier:.4f} (expected: 0.0)")

    def test_brier_null_maximized_at_balanced(self):
        """Test that Brier_null is maximized when dataset is balanced (p = 0.5)."""
        # Test various prevalence values
        prevalences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        brier_nulls = []

        for p in prevalences:
            brier_null = p * (1 - p)
            brier_nulls.append(brier_null)

        # Maximum should be at p = 0.5
        max_brier_null = max(brier_nulls)
        max_index = brier_nulls.index(max_brier_null)

        self.assertEqual(prevalences[max_index], 0.5)
        self.assertAlmostEqual(max_brier_null, 0.25, places=6)

        print(f"\n[OK] Brier_null maximized at balanced dataset (p = 0.5)")
        print(f"     Maximum Brier_null: {max_brier_null:.4f} at p = {prevalences[max_index]}")

    def test_numerical_example_from_documentation(self):
        """Test the exact numerical example from the documentation."""
        # Example from BRIER_METRICS_IMPLEMENTATION.md
        labels = np.array([0, 0, 1, 1, 1])
        prevalence = labels.mean()  # 0.6

        # Brier_null
        brier_null = compute_brier_null(labels)
        self.assertAlmostEqual(prevalence, 0.6, places=6)
        self.assertAlmostEqual(brier_null, 0.24, places=6)

        # Good model
        predictions_good = np.array([0.1, 0.1, 0.9, 0.9, 0.9])
        brier_good = brier_score_loss(labels, predictions_good)
        scaled_brier_good = compute_scaled_brier(brier_good, labels)

        # Expected: Brier_good = mean([0.01, 0.01, 0.01, 0.01, 0.01]) = 0.01
        self.assertAlmostEqual(brier_good, 0.01, places=6)
        # Expected: Scaled Brier = 1 - (0.01 / 0.24) = 1 - 0.042 = 0.958
        expected_scaled = 1 - (0.01 / 0.24)
        self.assertAlmostEqual(scaled_brier_good, expected_scaled, places=3)

        print(f"\n[OK] Numerical example from documentation verified")
        print(f"     Prevalence: {prevalence:.2f}")
        print(f"     Brier_null: {brier_null:.4f}")
        print(f"     Brier (good model): {brier_good:.4f}")
        print(f"     Scaled Brier: {scaled_brier_good:.4f} (expected: {expected_scaled:.4f})")


def main():
    """Run all tests and display summary."""
    print("=" * 80)
    print("BRIER METRICS TEST SUITE")
    print("=" * 80)
    print("\nVerifying:")
    print("  1. Prevalence: p = mean(y)")
    print("  2. Brier_null = p * (1 - p)")
    print("  3. Scaled Brier = 1 - (Brier / Brier_null)")
    print("  4. Edge cases and numerical examples")
    print("\n" + "=" * 80)

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBrierMetrics)
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
        print("  - Prevalence computed correctly: p = mean(y)")
        print("  - Brier_null formula verified: p * (1 - p)")
        print("  - Scaled Brier formula verified: 1 - (Brier / Brier_null)")
        print("  - Edge cases handled (all zeros, all ones)")
        print("  - Brier_null maximized at p = 0.5 (balanced)")
        print("  - compute_metrics() returns all 5 metrics")
    else:
        print("\n[FAIL] Some tests failed")

    print("=" * 80)


if __name__ == "__main__":
    main()
