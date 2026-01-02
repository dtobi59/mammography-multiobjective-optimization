"""
Verification script to demonstrate imports work without 'pip install -e .'

This script should work immediately after cloning the repository,
without any package installation beyond requirements.txt.
"""

print("Testing imports without package installation...")
print("=" * 60)

try:
    # Test config import
    import config
    print("[OK] config imported successfully")

    # Test data module
    from data.dataset import MammographyDataset, create_train_val_split
    from data.augmentation import IntensityAugmentation
    print("[OK] data module imported successfully")

    # Test models module
    from models.resnet import ResNet50WithPartialFineTuning
    print("[OK] models module imported successfully")

    # Test training module
    from training.trainer import Trainer
    from training.metrics import compute_metrics
    print("[OK] training module imported successfully")

    # Test optimization module
    from optimization.problem import BreastCancerOptimizationProblem
    from optimization.nsga3_runner import NSGA3Runner, load_metadata
    print("[OK] optimization module imported successfully")

    # Test utils module
    from utils.seed import set_all_seeds
    from utils.noisy_or import noisy_or_aggregation
    print("[OK] utils module imported successfully")

    print("=" * 60)
    print("\n[SUCCESS] All imports work without 'pip install -e .'")
    print("\nThe import error has been fixed!")
    print("\nYou can now:")
    print("  1. Run scripts directly: python optimization/nsga3_runner.py")
    print("  2. Run tests: python test_correctness.py")
    print("  3. Use Colab notebook: just clone and run!")
    print("\nNo package installation required beyond:")
    print("  pip install -r requirements.txt")

except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}")
    print("\nThis shouldn't happen! Please report this issue.")
    exit(1)
