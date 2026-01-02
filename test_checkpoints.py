"""
Test checkpoint functionality for NSGA-III optimization.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_checkpoint_callback():
    """Test CheckpointCallback creation and directory setup."""
    from optimization.nsga3_runner import CheckpointCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create callback
        callback = CheckpointCallback(
            checkpoint_dir=tmpdir,
            save_frequency=2
        )

        # Verify directory was created
        assert Path(tmpdir).exists(), "Checkpoint directory should exist"
        assert callback.save_frequency == 2, "Save frequency should be 2"
        assert callback.checkpoint_dir == Path(tmpdir), "Checkpoint dir should match"

        print("[PASS] CheckpointCallback creation")

def test_nsga3_runner_checkpoint_setup():
    """Test NSGA3Runner checkpoint directory setup."""
    from optimization.nsga3_runner import NSGA3Runner
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy metadata
        dummy_metadata = pd.DataFrame({
            "image_id": ["img1", "img2"],
            "patient_id": ["p1", "p2"],
            "breast_id": ["b1", "b2"],
            "view": ["CC", "MLO"],
            "label": [0, 1],
            "image_path": ["path1", "path2"]
        })

        output_dir = os.path.join(tmpdir, "output")
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")

        # Create runner
        runner = NSGA3Runner(
            train_metadata=dummy_metadata,
            val_metadata=dummy_metadata,
            image_dir=tmpdir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            save_frequency=5
        )

        # Verify checkpoint directories were created
        assert os.path.exists(output_dir), "Output dir should exist"
        assert os.path.exists(checkpoint_dir), "Checkpoint dir should exist"
        assert os.path.exists(runner.opt_checkpoint_dir), "Optimization checkpoint dir should exist"
        assert runner.save_frequency == 5, "Save frequency should be 5"
        assert runner.evaluation_history == [], "Evaluation history should be empty"

        print("[PASS] NSGA3Runner checkpoint setup")

def test_list_checkpoints():
    """Test checkpoint listing functionality."""
    from optimization.nsga3_runner import NSGA3Runner
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy metadata
        dummy_metadata = pd.DataFrame({
            "image_id": ["img1"],
            "patient_id": ["p1"],
            "breast_id": ["b1"],
            "view": ["CC"],
            "label": [0],
            "image_path": ["path1"]
        })

        output_dir = os.path.join(tmpdir, "output")

        # Create runner
        runner = NSGA3Runner(
            train_metadata=dummy_metadata,
            val_metadata=dummy_metadata,
            image_dir=tmpdir,
            output_dir=output_dir
        )

        # Initially no checkpoints
        checkpoints = runner.list_checkpoints()
        assert checkpoints == [], "Should have no checkpoints initially"

        # Create dummy checkpoint files
        checkpoint_dir = Path(runner.opt_checkpoint_dir)
        (checkpoint_dir / "checkpoint_gen_0001.pkl").touch()
        (checkpoint_dir / "checkpoint_gen_0002.pkl").touch()
        (checkpoint_dir / "checkpoint_gen_0005.pkl").touch()

        # List checkpoints
        checkpoints = runner.list_checkpoints()
        assert len(checkpoints) == 3, "Should have 3 checkpoints"
        assert checkpoints[0].name == "checkpoint_gen_0001.pkl", "First checkpoint should be gen 1"
        assert checkpoints[-1].name == "checkpoint_gen_0005.pkl", "Last checkpoint should be gen 5"

        print("[PASS] List checkpoints")

def test_checkpoint_callback_dataframe_creation():
    """Test _create_pareto_dataframe method."""
    from optimization.nsga3_runner import CheckpointCallback
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        callback = CheckpointCallback(tmpdir)

        # Test with None
        df = callback._create_pareto_dataframe(None, None)
        assert len(df) == 0, "Should return empty DataFrame for None inputs"

        # Test with valid data
        X = np.array([
            [-3.0, -4.0, 0.2, 0.5, 0.3],
            [-2.5, -3.5, 0.3, 0.6, 0.4]
        ])
        F = np.array([
            [-0.8, -0.7, 0.15, 0.05],
            [-0.85, -0.75, 0.12, 0.03]
        ])

        df = callback._create_pareto_dataframe(X, F)
        assert len(df) == 2, "Should have 2 rows"
        assert "solution_id" in df.columns, "Should have solution_id column"
        assert "log_lr" in df.columns, "Should have log_lr column"
        assert "obj_neg_pr_auc" in df.columns, "Should have obj_neg_pr_auc column"
        assert df.iloc[0]["log_lr"] == -3.0, "First solution log_lr should be -3.0"

        print("[PASS] CheckpointCallback DataFrame creation")

def main():
    """Run all checkpoint tests."""
    print("=" * 60)
    print("CHECKPOINT FUNCTIONALITY TESTS")
    print("=" * 60)
    print()

    tests = [
        ("CheckpointCallback creation", test_checkpoint_callback),
        ("NSGA3Runner checkpoint setup", test_nsga3_runner_checkpoint_setup),
        ("List checkpoints", test_list_checkpoints),
        ("CheckpointCallback DataFrame creation", test_checkpoint_callback_dataframe_creation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed} ({100 * passed / (passed + failed):.1f}%)")
    print(f"Failed: {failed}")
    print()

    if failed == 0:
        print("[SUCCESS] ALL CHECKPOINT TESTS PASSED!")
        return 0
    else:
        print("[FAILURE] Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
