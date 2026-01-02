"""
Comprehensive correctness tests for the implementation.
Tests all components without requiring actual data.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Test counters
tests_passed = 0
tests_failed = 0


def test_result(name: str, passed: bool, error_msg: str = ""):
    """Record test result."""
    global tests_passed, tests_failed
    if passed:
        print(f"[PASS] {name}")
        tests_passed += 1
    else:
        print(f"[FAIL] {name}")
        if error_msg:
            print(f"  Error: {error_msg}")
        tests_failed += 1


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Import Tests")
    print("="*60)

    try:
        import config
        test_result("Import config", True)
    except Exception as e:
        test_result("Import config", False, str(e))

    try:
        from utils.seed import set_all_seeds
        test_result("Import utils.seed", True)
    except Exception as e:
        test_result("Import utils.seed", False, str(e))

    try:
        from utils.noisy_or import noisy_or_aggregation, aggregate_to_breast_level
        test_result("Import utils.noisy_or", True)
    except Exception as e:
        test_result("Import utils.noisy_or", False, str(e))

    try:
        from data.augmentation import IntensityAugmentation, RobustnessPerturbation
        test_result("Import data.augmentation", True)
    except Exception as e:
        test_result("Import data.augmentation", False, str(e))

    try:
        from data.dataset import MammographyDataset, create_train_val_split
        test_result("Import data.dataset", True)
    except Exception as e:
        test_result("Import data.dataset", False, str(e))

    try:
        from models.resnet import ResNet50WithPartialFineTuning
        test_result("Import models.resnet", True)
    except Exception as e:
        test_result("Import models.resnet", False, str(e))

    try:
        from training.metrics import compute_metrics, compute_robustness_degradation, find_optimal_threshold
        test_result("Import training.metrics", True)
    except Exception as e:
        test_result("Import training.metrics", False, str(e))

    try:
        from training.trainer import Trainer, EarlyStopping
        test_result("Import training.trainer", True)
    except Exception as e:
        test_result("Import training.trainer", False, str(e))

    try:
        from training.robustness import RobustnessEvaluator
        test_result("Import training.robustness", True)
    except Exception as e:
        test_result("Import training.robustness", False, str(e))

    try:
        from optimization.problem import BreastCancerOptimizationProblem
        test_result("Import optimization.problem", True)
    except Exception as e:
        test_result("Import optimization.problem", False, str(e))


def test_noisy_or():
    """Test Noisy OR aggregation."""
    print("\n" + "="*60)
    print("TEST 2: Noisy OR Aggregation")
    print("="*60)

    from utils.noisy_or import noisy_or_aggregation, aggregate_to_breast_level

    # Test case 1: Both probabilities zero
    result = noisy_or_aggregation(0.0, 0.0)
    expected = 0.0
    test_result(
        "Noisy OR: p_cc=0, p_mlo=0",
        abs(result - expected) < 1e-6,
        f"Expected {expected}, got {result}"
    )

    # Test case 2: One probability zero
    result = noisy_or_aggregation(0.5, 0.0)
    expected = 0.5
    test_result(
        "Noisy OR: p_cc=0.5, p_mlo=0",
        abs(result - expected) < 1e-6,
        f"Expected {expected}, got {result}"
    )

    # Test case 3: Both probabilities non-zero
    result = noisy_or_aggregation(0.3, 0.4)
    expected = 1 - (1 - 0.3) * (1 - 0.4)  # 1 - 0.7 * 0.6 = 1 - 0.42 = 0.58
    test_result(
        "Noisy OR: p_cc=0.3, p_mlo=0.4",
        abs(result - expected) < 1e-6,
        f"Expected {expected}, got {result}"
    )

    # Test case 4: Both probabilities one
    result = noisy_or_aggregation(1.0, 1.0)
    expected = 1.0
    test_result(
        "Noisy OR: p_cc=1, p_mlo=1",
        abs(result - expected) < 1e-6,
        f"Expected {expected}, got {result}"
    )

    # Test breast-level aggregation
    metadata = pd.DataFrame({
        "image_id": ["img1", "img2", "img3", "img4"],
        "patient_id": ["p1", "p1", "p2", "p2"],
        "breast_id": ["b1", "b1", "b2", "b2"],
        "view": ["CC", "MLO", "CC", "MLO"],
        "label": [0, 0, 1, 1],
    })

    predictions = {
        "img1": 0.2,  # b1 CC
        "img2": 0.3,  # b1 MLO
        "img3": 0.6,  # b2 CC
        "img4": 0.7,  # b2 MLO
    }

    breast_preds, breast_labels = aggregate_to_breast_level(predictions, metadata)

    # Expected: b1 = 1 - (1-0.2)*(1-0.3) = 1 - 0.8*0.7 = 0.44
    #           b2 = 1 - (1-0.6)*(1-0.7) = 1 - 0.4*0.3 = 0.88
    expected_b1 = 1 - (1 - 0.2) * (1 - 0.3)
    expected_b2 = 1 - (1 - 0.6) * (1 - 0.7)

    test_result(
        "Breast-level aggregation: shape",
        len(breast_preds) == 2 and len(breast_labels) == 2,
        f"Expected 2 breasts, got {len(breast_preds)}"
    )

    test_result(
        "Breast-level aggregation: labels",
        np.array_equal(sorted(breast_labels), [0, 1]),
        f"Expected [0, 1], got {sorted(breast_labels)}"
    )

    # Check predictions are close to expected (order might vary)
    pred_set = set(np.round(breast_preds, 4))
    expected_set = {round(expected_b1, 4), round(expected_b2, 4)}
    test_result(
        "Breast-level aggregation: predictions",
        pred_set == expected_set,
        f"Expected {expected_set}, got {pred_set}"
    )


def test_augmentation():
    """Test augmentation pipeline."""
    print("\n" + "="*60)
    print("TEST 3: Augmentation Pipeline")
    print("="*60)

    from utils.seed import set_all_seeds
    from data.augmentation import IntensityAugmentation, RobustnessPerturbation

    set_all_seeds(42)

    # Test zero strength (no augmentation)
    aug_zero = IntensityAugmentation(strength=0.0)
    image = torch.rand(3, 224, 224)
    aug_image = aug_zero(image.clone())

    test_result(
        "Augmentation strength=0: no change",
        torch.allclose(image, aug_image),
        f"Image changed with strength=0"
    )

    # Test non-zero strength (should change)
    aug_half = IntensityAugmentation(strength=0.5)
    set_all_seeds(42)
    image = torch.rand(3, 224, 224)
    aug_image = aug_half(image.clone())

    test_result(
        "Augmentation strength=0.5: changes image",
        not torch.allclose(image, aug_image),
        f"Image unchanged with strength=0.5"
    )

    # Test output range [0, 1]
    test_result(
        "Augmentation: output in [0,1]",
        aug_image.min() >= 0.0 and aug_image.max() <= 1.0,
        f"Output range [{aug_image.min():.3f}, {aug_image.max():.3f}]"
    )

    # Test robustness perturbation
    perturb = RobustnessPerturbation()
    image = torch.rand(3, 224, 224)
    perturbed = perturb(image.clone(), seed=42)

    test_result(
        "Robustness perturbation: changes image",
        not torch.allclose(image, perturbed),
        f"Image unchanged after perturbation"
    )

    test_result(
        "Robustness perturbation: output in [0,1]",
        perturbed.min() >= 0.0 and perturbed.max() <= 1.0,
        f"Output range [{perturbed.min():.3f}, {perturbed.max():.3f}]"
    )

    # Test reproducibility with seed
    perturbed2 = perturb(image.clone(), seed=42)
    test_result(
        "Robustness perturbation: reproducible with seed",
        torch.allclose(perturbed, perturbed2),
        f"Different results with same seed"
    )


def test_model():
    """Test model architecture."""
    print("\n" + "="*60)
    print("TEST 4: Model Architecture")
    print("="*60)

    from models.resnet import ResNet50WithPartialFineTuning

    # Test model creation with different unfreeze fractions
    for unfreeze_frac in [0.0, 0.3, 0.5, 1.0]:
        try:
            model = ResNet50WithPartialFineTuning(
                unfreeze_fraction=unfreeze_frac,
                dropout_rate=0.2,
                pretrained=False,  # Faster without downloading weights
            )
            test_result(
                f"Model creation: unfreeze_frac={unfreeze_frac}",
                True
            )
        except Exception as e:
            test_result(
                f"Model creation: unfreeze_frac={unfreeze_frac}",
                False,
                str(e)
            )

    # Test forward pass
    model = ResNet50WithPartialFineTuning(
        unfreeze_fraction=0.5,
        dropout_rate=0.2,
        pretrained=False,
    )

    batch_size = 4
    input_tensor = torch.rand(batch_size, 3, 224, 224)

    try:
        output = model(input_tensor)
        test_result(
            "Model forward pass: runs",
            True
        )
    except Exception as e:
        test_result(
            "Model forward pass: runs",
            False,
            str(e)
        )
        return

    test_result(
        "Model output shape",
        output.shape == (batch_size,),
        f"Expected shape ({batch_size},), got {output.shape}"
    )

    test_result(
        "Model output range [0,1] (sigmoid)",
        output.min() >= 0.0 and output.max() <= 1.0,
        f"Output range [{output.min():.3f}, {output.max():.3f}]"
    )

    # Test parameter freezing
    trainable = model.get_trainable_params()
    frozen = model.get_frozen_params()
    total = trainable + frozen

    test_result(
        "Model has trainable params",
        trainable > 0,
        f"No trainable parameters"
    )

    test_result(
        "Model has frozen params with unfreeze_frac=0.5",
        frozen > 0,
        f"No frozen parameters"
    )

    # Test full fine-tuning
    model_full = ResNet50WithPartialFineTuning(
        unfreeze_fraction=1.0,
        dropout_rate=0.0,
        pretrained=False,
    )

    frozen_full = model_full.get_frozen_params()
    test_result(
        "Model unfreeze_frac=1.0: all unfrozen",
        frozen_full == 0,
        f"Has {frozen_full} frozen params with unfreeze_frac=1.0"
    )

    # Test freeze all
    model_freeze = ResNet50WithPartialFineTuning(
        unfreeze_fraction=0.0,
        dropout_rate=0.0,
        pretrained=False,
    )

    trainable_freeze = model_freeze.get_trainable_params()
    # Should have trainable params in classifier only
    test_result(
        "Model unfreeze_frac=0.0: classifier trainable",
        trainable_freeze > 0 and trainable_freeze < total,
        f"Trainable params: {trainable_freeze}/{total}"
    )


def test_metrics():
    """Test metric computation."""
    print("\n" + "="*60)
    print("TEST 5: Metrics Computation")
    print("="*60)

    from training.metrics import (
        compute_metrics,
        compute_robustness_degradation,
        find_optimal_threshold,
        compute_sensitivity_specificity,
    )

    # Create test data
    np.random.seed(42)
    n_samples = 100
    labels = np.random.randint(0, 2, n_samples)
    predictions = np.random.rand(n_samples)

    # Test compute_metrics
    try:
        metrics = compute_metrics(predictions, labels)
        test_result("Metrics computation: runs", True)

        required_keys = ["pr_auc", "auroc", "brier"]
        has_all_keys = all(k in metrics for k in required_keys)
        test_result(
            "Metrics: has all required keys",
            has_all_keys,
            f"Missing keys: {set(required_keys) - set(metrics.keys())}"
        )

        # Check metric ranges
        test_result(
            "PR-AUC in [0,1]",
            0 <= metrics["pr_auc"] <= 1,
            f"PR-AUC = {metrics['pr_auc']}"
        )

        test_result(
            "AUROC in [0,1]",
            0 <= metrics["auroc"] <= 1,
            f"AUROC = {metrics['auroc']}"
        )

        test_result(
            "Brier in [0,1]",
            0 <= metrics["brier"] <= 1,
            f"Brier = {metrics['brier']}"
        )

    except Exception as e:
        test_result("Metrics computation: runs", False, str(e))

    # Test robustness degradation
    predictions2 = predictions + np.random.randn(n_samples) * 0.1
    predictions2 = np.clip(predictions2, 0, 1)

    try:
        degradation = compute_robustness_degradation(
            predictions, predictions2, labels
        )
        test_result("Robustness degradation: runs", True)

        test_result(
            "Robustness degradation: is float",
            isinstance(degradation, (float, np.floating)),
            f"Type: {type(degradation)}"
        )
    except Exception as e:
        test_result("Robustness degradation: runs", False, str(e))

    # Test threshold finding
    try:
        threshold = find_optimal_threshold(predictions, labels)
        test_result("Threshold finding: runs", True)

        test_result(
            "Threshold in [0,1]",
            0 <= threshold <= 1,
            f"Threshold = {threshold}"
        )
    except Exception as e:
        test_result("Threshold finding: runs", False, str(e))

    # Test sensitivity/specificity
    try:
        sens, spec = compute_sensitivity_specificity(predictions, labels, 0.5)
        test_result("Sensitivity/Specificity: runs", True)

        test_result(
            "Sensitivity in [0,1]",
            0 <= sens <= 1,
            f"Sensitivity = {sens}"
        )

        test_result(
            "Specificity in [0,1]",
            0 <= spec <= 1,
            f"Specificity = {spec}"
        )
    except Exception as e:
        test_result("Sensitivity/Specificity: runs", False, str(e))


def test_data_splitting():
    """Test patient-wise data splitting."""
    print("\n" + "="*60)
    print("TEST 6: Data Splitting")
    print("="*60)

    from data.dataset import create_train_val_split

    # Create dummy metadata
    metadata = pd.DataFrame({
        "image_id": [f"img{i}" for i in range(100)],
        "patient_id": [f"p{i//5}" for i in range(100)],  # 20 patients, 5 images each
        "breast_id": [f"b{i//5}" for i in range(100)],
        "view": ["CC" if i % 2 == 0 else "MLO" for i in range(100)],
        "label": [i % 2 for i in range(100)],
        "image_path": [f"img{i}.png" for i in range(100)],
    })

    train_meta, val_meta = create_train_val_split(metadata, train_ratio=0.8, random_seed=42)

    test_result(
        "Data split: creates train and val",
        len(train_meta) > 0 and len(val_meta) > 0,
        f"Train: {len(train_meta)}, Val: {len(val_meta)}"
    )

    # Check patient-wise splitting (no patient in both sets)
    train_patients = set(train_meta["patient_id"].unique())
    val_patients = set(val_meta["patient_id"].unique())

    test_result(
        "Data split: patient-wise (no overlap)",
        len(train_patients & val_patients) == 0,
        f"Overlapping patients: {train_patients & val_patients}"
    )

    # Check approximately 80/20 split
    total_patients = len(metadata["patient_id"].unique())
    expected_train = int(total_patients * 0.8)

    test_result(
        "Data split: correct ratio",
        abs(len(train_patients) - expected_train) <= 1,
        f"Expected ~{expected_train} train patients, got {len(train_patients)}"
    )

    # Test reproducibility
    train_meta2, val_meta2 = create_train_val_split(metadata, train_ratio=0.8, random_seed=42)

    test_result(
        "Data split: reproducible with seed",
        train_meta.equals(train_meta2) and val_meta.equals(val_meta2),
        "Different splits with same seed"
    )


def test_early_stopping():
    """Test early stopping logic."""
    print("\n" + "="*60)
    print("TEST 7: Early Stopping")
    print("="*60)

    from training.trainer import EarlyStopping

    # Test maximization mode
    early_stop = EarlyStopping(patience=3, mode="max", delta=0.01)

    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]  # Improvement then decline

    for epoch, score in enumerate(scores):
        should_stop = early_stop(score, epoch)

        if epoch < 3:
            test_result(
                f"Early stopping epoch {epoch}: not triggered",
                not should_stop,
                f"Stopped too early at epoch {epoch}"
            )

    test_result(
        "Early stopping: triggers after patience",
        early_stop.early_stop,
        "Early stopping not triggered"
    )

    test_result(
        "Early stopping: best epoch is 2",
        early_stop.best_epoch == 2,
        f"Best epoch: {early_stop.best_epoch}"
    )

    # Test minimization mode
    early_stop_min = EarlyStopping(patience=2, mode="min", delta=0.01)

    scores_min = [0.5, 0.4, 0.45, 0.46]  # Improvement then plateau

    for epoch, score in enumerate(scores_min):
        should_stop = early_stop_min(score, epoch)

    test_result(
        "Early stopping (min mode): triggers",
        early_stop_min.early_stop,
        "Early stopping not triggered in min mode"
    )


def test_pymoo_problem():
    """Test pymoo Problem definition."""
    print("\n" + "="*60)
    print("TEST 8: pymoo Problem Definition")
    print("="*60)

    from optimization.problem import BreastCancerOptimizationProblem
    import config

    # Create dummy metadata
    n_samples = 20
    metadata = pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n_samples)],
        "patient_id": [f"p{i//4}" for i in range(n_samples)],
        "breast_id": [f"b{i//2}" for i in range(n_samples)],
        "view": ["CC" if i % 2 == 0 else "MLO" for i in range(n_samples)],
        "label": [i % 2 for i in range(n_samples)],
        "image_path": [f"img{i}.png" for i in range(n_samples)],
    })

    train_meta = metadata[:16]
    val_meta = metadata[16:]

    try:
        problem = BreastCancerOptimizationProblem(
            train_metadata=train_meta,
            val_metadata=val_meta,
            image_dir="/dummy/path",
            checkpoint_dir="./test_checkpoints",
        )
        test_result("Problem creation: runs", True)
    except Exception as e:
        test_result("Problem creation: runs", False, str(e))
        return

    test_result(
        "Problem: correct number of variables",
        problem.n_var == 5,
        f"Expected 5 variables, got {problem.n_var}"
    )

    test_result(
        "Problem: correct number of objectives",
        problem.n_obj == 4,
        f"Expected 4 objectives, got {problem.n_obj}"
    )

    test_result(
        "Problem: no constraints",
        problem.n_constr == 0,
        f"Expected 0 constraints, got {problem.n_constr}"
    )

    # Test hyperparameter decoding
    x_test = np.array([
        -3.0,  # log10(learning_rate) = -3 -> lr = 0.001
        -4.0,  # log10(weight_decay) = -4 -> wd = 0.0001
        0.2,   # dropout_rate
        0.5,   # augmentation_strength
        0.3,   # unfreeze_fraction
    ])

    hparams = problem._decode_hyperparameters(x_test)

    test_result(
        "Hyperparameter decoding: learning_rate",
        abs(hparams["learning_rate"] - 0.001) < 1e-6,
        f"Expected 0.001, got {hparams['learning_rate']}"
    )

    test_result(
        "Hyperparameter decoding: weight_decay",
        abs(hparams["weight_decay"] - 0.0001) < 1e-6,
        f"Expected 0.0001, got {hparams['weight_decay']}"
    )

    test_result(
        "Hyperparameter decoding: dropout_rate",
        abs(hparams["dropout_rate"] - 0.2) < 1e-6,
        f"Expected 0.2, got {hparams['dropout_rate']}"
    )


def test_config():
    """Test configuration values."""
    print("\n" + "="*60)
    print("TEST 9: Configuration")
    print("="*60)

    import config

    # Check required configuration exists
    required_attrs = [
        "RANDOM_SEED",
        "TRAIN_VAL_SPLIT",
        "IMAGE_SIZE",
        "BATCH_SIZE",
        "MAX_EPOCHS",
        "EARLY_STOPPING_PATIENCE",
        "HYPERPARAMETER_BOUNDS",
        "NSGA3_CONFIG",
    ]

    for attr in required_attrs:
        test_result(
            f"Config has {attr}",
            hasattr(config, attr),
            f"Missing configuration: {attr}"
        )

    # Check hyperparameter bounds
    if hasattr(config, "HYPERPARAMETER_BOUNDS"):
        required_hparams = [
            "learning_rate",
            "weight_decay",
            "dropout_rate",
            "augmentation_strength",
            "unfreeze_fraction",
        ]

        for hp in required_hparams:
            test_result(
                f"Config hyperparameter bounds: {hp}",
                hp in config.HYPERPARAMETER_BOUNDS,
                f"Missing hyperparameter: {hp}"
            )

    # Check NSGA3 config
    if hasattr(config, "NSGA3_CONFIG"):
        test_result(
            "NSGA3 config: n_objectives = 4",
            config.NSGA3_CONFIG.get("n_objectives") == 4,
            f"Expected 4 objectives, got {config.NSGA3_CONFIG.get('n_objectives')}"
        )


def test_seed_reproducibility():
    """Test random seed setting."""
    print("\n" + "="*60)
    print("TEST 10: Seed Reproducibility")
    print("="*60)

    from utils.seed import set_all_seeds

    # Test NumPy reproducibility
    set_all_seeds(42)
    rand1 = np.random.rand(10)

    set_all_seeds(42)
    rand2 = np.random.rand(10)

    test_result(
        "NumPy reproducibility with seed",
        np.allclose(rand1, rand2),
        "Different NumPy random values with same seed"
    )

    # Test PyTorch reproducibility
    set_all_seeds(42)
    torch_rand1 = torch.rand(10)

    set_all_seeds(42)
    torch_rand2 = torch.rand(10)

    test_result(
        "PyTorch reproducibility with seed",
        torch.allclose(torch_rand1, torch_rand2),
        "Different PyTorch random values with same seed"
    )


def print_summary():
    """Print test summary."""
    global tests_passed, tests_failed

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_tests = tests_passed + tests_failed
    pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"Total tests: {total_tests}")
    print(f"Passed: {tests_passed} ({pass_rate:.1f}%)")
    print(f"Failed: {tests_failed}")

    if tests_failed == 0:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n[FAILURE] {tests_failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE CORRECTNESS TESTS")
    print("="*60)
    print("Testing implementation without requiring real data...")

    # Run all tests
    test_imports()
    test_noisy_or()
    test_augmentation()
    test_model()
    test_metrics()
    test_data_splitting()
    test_early_stopping()
    test_pymoo_problem()
    test_config()
    test_seed_reproducibility()

    # Print summary and exit
    exit_code = print_summary()
    sys.exit(exit_code)
