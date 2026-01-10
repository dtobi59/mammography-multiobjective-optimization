"""
Comprehensive unit tests to verify ZERO patient leakage between train and validation sets.

This test suite verifies:
1. No patient appears in both train and validation sets
2. All patients are accounted for (complete partition)
3. Split ratios are correct
4. Reproducibility with fixed seeds
5. Manifest file integrity
6. Edge cases (single patient, all train, all val)

CRITICAL: These tests MUST pass. Any failure indicates data leakage.
"""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd
import numpy as np
import unittest
from glob import glob
import config
from data.dataset import create_train_val_split, create_stratified_subsample
from data.parsers import VinDrMammoParser


class TestPatientLeakage(unittest.TestCase):
    """Test suite to verify zero patient leakage."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic metadata for testing
        self.test_metadata = self._create_test_metadata(
            n_patients=100,
            images_per_patient=5
        )

    def _create_test_metadata(self, n_patients: int, images_per_patient: int) -> pd.DataFrame:
        """Create synthetic metadata for testing."""
        rows = []
        for patient_id in range(n_patients):
            for image_id in range(images_per_patient):
                rows.append({
                    "image_id": f"img_{patient_id}_{image_id}",
                    "patient_id": f"patient_{patient_id:04d}",
                    "breast_id": f"patient_{patient_id:04d}_{'R' if image_id % 2 == 0 else 'L'}",
                    "laterality": "R" if image_id % 2 == 0 else "L",
                    "view": "CC" if image_id < images_per_patient // 2 else "MLO",
                    "label": patient_id % 3 == 0,  # ~33% malignant
                    "image_path": f"patient_{patient_id:04d}/img_{patient_id}_{image_id}.png",
                    "birads_original": "BI-RADS 4" if patient_id % 3 == 0 else "BI-RADS 2",
                })
        return pd.DataFrame(rows)

    # =========================================================================
    # TEST 1: Core Split Function - No Overlap
    # =========================================================================

    def test_no_patient_overlap_basic(self):
        """Test that train and val sets have zero patient overlap."""
        train_meta, val_meta = create_train_val_split(
            self.test_metadata,
            train_ratio=0.8,
            random_seed=42
        )

        train_patients = set(train_meta["patient_id"].unique())
        val_patients = set(val_meta["patient_id"].unique())
        overlap = train_patients & val_patients

        self.assertEqual(
            len(overlap), 0,
            f"CRITICAL FAILURE: {len(overlap)} patients in both train and val: {overlap}"
        )

    # =========================================================================
    # TEST 2: Complete Partition
    # =========================================================================

    def test_complete_partition(self):
        """Test that all patients are accounted for (union equals original)."""
        train_meta, val_meta = create_train_val_split(
            self.test_metadata,
            train_ratio=0.8,
            random_seed=42
        )

        train_patients = set(train_meta["patient_id"].unique())
        val_patients = set(val_meta["patient_id"].unique())
        all_patients = set(self.test_metadata["patient_id"].unique())

        union = train_patients | val_patients

        self.assertEqual(
            union, all_patients,
            f"Partition incomplete: {len(all_patients - union)} patients missing"
        )

    # =========================================================================
    # TEST 3: Correct Split Ratio
    # =========================================================================

    def test_split_ratio(self):
        """Test that split ratio is approximately correct."""
        train_meta, val_meta = create_train_val_split(
            self.test_metadata,
            train_ratio=0.8,
            random_seed=42
        )

        total_patients = self.test_metadata["patient_id"].nunique()
        train_patients = train_meta["patient_id"].nunique()
        expected_train = int(total_patients * 0.8)

        # Allow ±1 patient due to integer rounding
        self.assertAlmostEqual(
            train_patients, expected_train, delta=1,
            msg=f"Split ratio incorrect: expected ~{expected_train} train patients, got {train_patients}"
        )

    # =========================================================================
    # TEST 4: Reproducibility with Fixed Seed
    # =========================================================================

    def test_reproducibility(self):
        """Test that same seed produces identical splits."""
        train1, val1 = create_train_val_split(self.test_metadata, random_seed=42)
        train2, val2 = create_train_val_split(self.test_metadata, random_seed=42)

        train1_patients = set(train1["patient_id"].unique())
        train2_patients = set(train2["patient_id"].unique())

        self.assertEqual(
            train1_patients, train2_patients,
            "Same seed produced different train sets"
        )

    # =========================================================================
    # TEST 5: Different Seeds Produce Different Splits
    # =========================================================================

    def test_different_seeds(self):
        """Test that different seeds produce different splits."""
        train_seed1, _ = create_train_val_split(self.test_metadata, random_seed=42)
        train_seed2, _ = create_train_val_split(self.test_metadata, random_seed=123)

        train1_patients = set(train_seed1["patient_id"].unique())
        train2_patients = set(train_seed2["patient_id"].unique())

        # Different seeds should produce different train sets
        # (unless extremely unlikely collision)
        self.assertNotEqual(
            train1_patients, train2_patients,
            "Different seeds produced identical splits (possible but unlikely)"
        )

    # =========================================================================
    # TEST 6: Manifest File Integrity
    # =========================================================================

    def test_manifest_file_no_leakage(self):
        """Test that saved manifest files have no patient leakage."""
        # Find all manifest files
        manifest_pattern = str(Path(__file__).parent / "manifests" / "*.csv")
        manifest_files = glob(manifest_pattern)

        if len(manifest_files) == 0:
            self.skipTest("No manifest files found to test")

        for manifest_path in manifest_files:
            with self.subTest(manifest=Path(manifest_path).name):
                manifest = pd.read_csv(manifest_path)

                # Check that manifest has required columns
                self.assertIn("patient_id", manifest.columns)
                self.assertIn("split", manifest.columns)

                # Split by train/val
                train = manifest[manifest["split"] == "train"]
                val = manifest[manifest["split"] == "val"]

                # Check no overlap
                train_patients = set(train["patient_id"].unique())
                val_patients = set(val["patient_id"].unique())
                overlap = train_patients & val_patients

                self.assertEqual(
                    len(overlap), 0,
                    f"LEAKAGE in {Path(manifest_path).name}: {len(overlap)} overlapping patients"
                )

    # =========================================================================
    # TEST 7: Stratified Subsample No Leakage
    # =========================================================================

    def test_stratified_subsample_no_leakage(self):
        """Test that stratified subsampling preserves no-leakage property."""
        # Add birads_original column for exclusion test
        metadata = self.test_metadata.copy()

        train_meta, val_meta, _ = create_stratified_subsample(
            metadata=metadata,
            target_total=200,
            target_malignant=50,
            target_benign=150,
            train_ratio=0.8,
            random_seed=42,
            output_dir="./test_manifests",
            exclude_birads_3=False,  # Don't exclude for this test
        )

        train_patients = set(train_meta["patient_id"].unique())
        val_patients = set(val_meta["patient_id"].unique())
        overlap = train_patients & val_patients

        self.assertEqual(
            len(overlap), 0,
            f"Stratified subsample has {len(overlap)} overlapping patients"
        )

    # =========================================================================
    # TEST 8: Edge Case - Single Patient
    # =========================================================================

    def test_single_patient_edge_case(self):
        """Test edge case with only one patient."""
        single_patient_data = self._create_test_metadata(n_patients=1, images_per_patient=4)

        train_meta, val_meta = create_train_val_split(
            single_patient_data,
            train_ratio=0.8,
            random_seed=42
        )

        # With 1 patient and 80/20 split, int(1 * 0.8) = 0 train patients
        # So all images go to validation (edge case)
        total_patients = single_patient_data["patient_id"].nunique()
        train_patients_count = train_meta["patient_id"].nunique()
        val_patients_count = val_meta["patient_id"].nunique()

        # Total should be preserved
        self.assertEqual(train_patients_count + val_patients_count, total_patients)

        # Check no overlap if both sets exist
        if len(train_meta) > 0 and len(val_meta) > 0:
            train_patients = set(train_meta["patient_id"].unique())
            val_patients = set(val_meta["patient_id"].unique())
            self.assertEqual(len(train_patients & val_patients), 0)

    # =========================================================================
    # TEST 9: Edge Case - Two Patients
    # =========================================================================

    def test_two_patients_edge_case(self):
        """Test edge case with only two patients."""
        two_patient_data = self._create_test_metadata(n_patients=2, images_per_patient=3)

        train_meta, val_meta = create_train_val_split(
            two_patient_data,
            train_ratio=0.5,
            random_seed=42
        )

        train_patients = set(train_meta["patient_id"].unique())
        val_patients = set(val_meta["patient_id"].unique())

        # Should be 1 patient in each set
        self.assertEqual(len(train_patients), 1)
        self.assertEqual(len(val_patients), 1)

        # No overlap
        self.assertEqual(len(train_patients & val_patients), 0)

    # =========================================================================
    # TEST 10: Edge Case - Extreme Ratios
    # =========================================================================

    def test_extreme_ratio_99_1(self):
        """Test extreme 99/1 split ratio."""
        train_meta, val_meta = create_train_val_split(
            self.test_metadata,
            train_ratio=0.99,
            random_seed=42
        )

        train_patients = set(train_meta["patient_id"].unique())
        val_patients = set(val_meta["patient_id"].unique())
        overlap = train_patients & val_patients

        self.assertEqual(len(overlap), 0, "Even extreme ratios should have no overlap")

    def test_extreme_ratio_1_99(self):
        """Test extreme 1/99 split ratio."""
        train_meta, val_meta = create_train_val_split(
            self.test_metadata,
            train_ratio=0.01,
            random_seed=42
        )

        train_patients = set(train_meta["patient_id"].unique())
        val_patients = set(val_meta["patient_id"].unique())
        overlap = train_patients & val_patients

        self.assertEqual(len(overlap), 0, "Even extreme ratios should have no overlap")

    # =========================================================================
    # TEST 11: Property-Based Test - Multiple Random Configurations
    # =========================================================================

    def test_property_based_multiple_configs(self):
        """Test no leakage holds across many random configurations."""
        test_configs = [
            (50, 10, 0.7, 123),    # 50 patients, 10 imgs/patient, 70/30, seed 123
            (200, 3, 0.8, 456),    # 200 patients, 3 imgs/patient, 80/20, seed 456
            (10, 20, 0.9, 789),    # 10 patients, 20 imgs/patient, 90/10, seed 789
            (1000, 2, 0.5, 42),    # 1000 patients, 2 imgs/patient, 50/50, seed 42
        ]

        for n_patients, imgs_per_patient, ratio, seed in test_configs:
            with self.subTest(n_patients=n_patients, ratio=ratio, seed=seed):
                metadata = self._create_test_metadata(n_patients, imgs_per_patient)
                train_meta, val_meta = create_train_val_split(
                    metadata,
                    train_ratio=ratio,
                    random_seed=seed
                )

                train_patients = set(train_meta["patient_id"].unique())
                val_patients = set(val_meta["patient_id"].unique())
                overlap = train_patients & val_patients

                self.assertEqual(
                    len(overlap), 0,
                    f"Config ({n_patients} patients, ratio={ratio}, seed={seed}) has {len(overlap)} overlapping patients"
                )

    # =========================================================================
    # TEST 12: Image-Level Verification
    # =========================================================================

    def test_all_images_from_same_patient_in_same_split(self):
        """Test that all images from the same patient are in the same split."""
        train_meta, val_meta = create_train_val_split(
            self.test_metadata,
            train_ratio=0.8,
            random_seed=42
        )

        # Check each patient in train
        for patient_id in train_meta["patient_id"].unique():
            patient_images_train = train_meta[train_meta["patient_id"] == patient_id]
            patient_images_val = val_meta[val_meta["patient_id"] == patient_id]

            self.assertEqual(
                len(patient_images_val), 0,
                f"Patient {patient_id} has images in both train and val"
            )

        # Check each patient in val
        for patient_id in val_meta["patient_id"].unique():
            patient_images_train = train_meta[train_meta["patient_id"] == patient_id]
            patient_images_val = val_meta[val_meta["patient_id"] == patient_id]

            self.assertEqual(
                len(patient_images_train), 0,
                f"Patient {patient_id} has images in both train and val"
            )

    # =========================================================================
    # TEST 13: Mathematical Invariants
    # =========================================================================

    def test_mathematical_invariants(self):
        """Test mathematical invariants of the split."""
        train_meta, val_meta = create_train_val_split(
            self.test_metadata,
            train_ratio=0.8,
            random_seed=42
        )

        train_patients = set(train_meta["patient_id"].unique())
        val_patients = set(val_meta["patient_id"].unique())
        all_patients = set(self.test_metadata["patient_id"].unique())

        # Invariant 1: Disjointness
        self.assertEqual(len(train_patients & val_patients), 0, "Sets must be disjoint")

        # Invariant 2: Completeness
        self.assertEqual(train_patients | val_patients, all_patients, "Sets must partition the universe")

        # Invariant 3: Cardinality
        self.assertEqual(
            len(train_patients) + len(val_patients),
            len(all_patients),
            "Cardinalities must sum to total"
        )


class TestManifestIntegrity(unittest.TestCase):
    """Additional tests specifically for manifest file integrity."""

    def test_all_manifests_in_directory(self):
        """Test all manifest files in the manifests directory."""
        manifests_dir = Path(__file__).parent / "manifests"

        if not manifests_dir.exists():
            self.skipTest("Manifests directory does not exist")

        manifest_files = list(manifests_dir.glob("*.csv"))

        if len(manifest_files) == 0:
            self.skipTest("No manifest files to test")

        print(f"\nTesting {len(manifest_files)} manifest files...")

        for manifest_path in manifest_files:
            with self.subTest(manifest=manifest_path.name):
                manifest = pd.read_csv(manifest_path)

                # Verify structure
                self.assertIn("patient_id", manifest.columns, f"{manifest_path.name}: Missing patient_id column")
                self.assertIn("split", manifest.columns, f"{manifest_path.name}: Missing split column")

                # Check splits
                train = manifest[manifest["split"] == "train"]
                val = manifest[manifest["split"] == "val"]

                self.assertGreater(len(train), 0, f"{manifest_path.name}: Empty train split")
                self.assertGreater(len(val), 0, f"{manifest_path.name}: Empty val split")

                # CRITICAL CHECK: No patient overlap
                train_patients = set(train["patient_id"].unique())
                val_patients = set(val["patient_id"].unique())
                overlap = train_patients & val_patients

                self.assertEqual(
                    len(overlap), 0,
                    f"CRITICAL: {manifest_path.name} has {len(overlap)} overlapping patients: {overlap}"
                )

                print(f"  [OK] {manifest_path.name}: {len(train_patients)} train, {len(val_patients)} val, 0 overlap")


def run_tests_with_summary():
    """Run tests and print summary."""
    print("=" * 80)
    print("PATIENT LEAKAGE TEST SUITE")
    print("=" * 80)
    print("\nRunning comprehensive tests to verify ZERO patient leakage...")
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestPatientLeakage))
    suite.addTests(loader.loadTestsFromTestCase(TestManifestIntegrity))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run:     {result.testsRun}")
    print(f"Successes:     {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")
    print(f"Skipped:       {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n[OK] ALL TESTS PASSED - ZERO PATIENT LEAKAGE CONFIRMED")
        print("=" * 80)
        return 0
    else:
        print("\n[FAIL] TESTS FAILED - REVIEW REQUIRED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = run_tests_with_summary()
    sys.exit(exit_code)
