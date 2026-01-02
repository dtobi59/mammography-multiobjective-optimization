"""
Test script to verify data preparation and setup.
"""

import os
import sys
import pandas as pd
import torch
from PIL import Image
import numpy as np


def check_metadata(metadata_path: str, dataset_name: str) -> bool:
    """Check if metadata file is properly formatted."""
    print(f"\n{'='*60}")
    print(f"Checking {dataset_name} metadata: {metadata_path}")
    print('='*60)

    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found: {metadata_path}")
        return False

    try:
        df = pd.read_csv(metadata_path)
        print(f"✓ Loaded metadata with {len(df)} rows")
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {e}")
        return False

    # Check required columns
    required_cols = ["image_id", "patient_id", "breast_id", "view", "label", "image_path"]
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return False

    print(f"✓ All required columns present: {required_cols}")

    # Check view values
    unique_views = df["view"].unique()
    print(f"✓ Unique views: {list(unique_views)}")
    if not all(v in ["CC", "MLO"] for v in unique_views):
        print(f"WARNING: Expected views to be 'CC' or 'MLO', found: {unique_views}")

    # Check labels
    unique_labels = df["label"].unique()
    print(f"✓ Unique labels: {list(unique_labels)}")
    if not all(l in [0, 1] for l in unique_labels):
        print(f"ERROR: Labels should be 0 or 1, found: {unique_labels}")
        return False

    # Check class distribution
    label_counts = df["label"].value_counts()
    print(f"✓ Class distribution:")
    print(f"  - Benign (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  - Malignant (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")

    # Check patient-wise statistics
    n_patients = df["patient_id"].nunique()
    n_breasts = df["breast_id"].nunique()
    print(f"✓ Unique patients: {n_patients}")
    print(f"✓ Unique breasts: {n_breasts}")

    # Check images per breast
    images_per_breast = df.groupby("breast_id").size()
    print(f"✓ Images per breast: min={images_per_breast.min()}, "
          f"max={images_per_breast.max()}, mean={images_per_breast.mean():.1f}")

    return True


def check_images(metadata_path: str, image_dir: str, n_samples: int = 5) -> bool:
    """Check if images exist and are properly formatted."""
    print(f"\n{'='*60}")
    print(f"Checking images in: {image_dir}")
    print('='*60)

    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        return False

    df = pd.read_csv(metadata_path)

    # Sample a few images
    sample_images = df.sample(min(n_samples, len(df)))["image_path"].tolist()

    print(f"Checking {len(sample_images)} sample images...")

    all_valid = True
    for img_path in sample_images:
        full_path = os.path.join(image_dir, img_path)

        if not os.path.exists(full_path):
            print(f"ERROR: Image not found: {full_path}")
            all_valid = False
            continue

        try:
            # Try to load image
            img = Image.open(full_path)
            img_array = np.array(img)

            print(f"✓ {img_path}: shape={img_array.shape}, dtype={img_array.dtype}, "
                  f"mode={img.mode}, size={img.size}")

            # Check if grayscale
            if img.mode != 'L' and img.mode != 'RGB':
                print(f"  WARNING: Expected grayscale (L) or RGB, got {img.mode}")

        except Exception as e:
            print(f"ERROR: Failed to load {full_path}: {e}")
            all_valid = False

    return all_valid


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    print(f"\n{'='*60}")
    print("Checking dependencies")
    print('='*60)

    dependencies = {
        "torch": torch,
        "torchvision": None,
        "numpy": np,
        "pandas": pd,
        "PIL": Image,
        "sklearn": None,
        "pymoo": None,
    }

    all_installed = True

    for name, module in dependencies.items():
        if module is None:
            try:
                __import__(name)
                print(f"✓ {name} installed")
            except ImportError:
                print(f"ERROR: {name} not installed")
                all_installed = False
        else:
            print(f"✓ {name} installed")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, will use CPU (training will be slow)")

    return all_installed


def test_data_loading():
    """Test data loading pipeline."""
    print(f"\n{'='*60}")
    print("Testing data loading pipeline")
    print('='*60)

    try:
        from data.dataset import create_train_val_split
        print("✓ Imported create_train_val_split")

        from data.augmentation import IntensityAugmentation, RobustnessPerturbation
        print("✓ Imported augmentation classes")

        from utils.noisy_or import noisy_or_aggregation
        print("✓ Imported noisy_or_aggregation")

        # Test Noisy OR
        p_breast = noisy_or_aggregation(0.3, 0.4)
        expected = 1 - (1 - 0.3) * (1 - 0.4)
        assert abs(p_breast - expected) < 1e-6, "Noisy OR computation error"
        print(f"✓ Noisy OR test passed: p_breast={p_breast:.4f}")

        # Test augmentation
        aug = IntensityAugmentation(strength=0.5)
        dummy_image = torch.rand(3, 224, 224)
        aug_image = aug(dummy_image)
        assert aug_image.shape == dummy_image.shape, "Augmentation shape mismatch"
        print(f"✓ Augmentation test passed")

        return True

    except Exception as e:
        print(f"ERROR: Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model creation."""
    print(f"\n{'='*60}")
    print("Testing model creation")
    print('='*60)

    try:
        from models import ResNet50WithPartialFineTuning

        model = ResNet50WithPartialFineTuning(
            unfreeze_fraction=0.3,
            dropout_rate=0.2,
            pretrained=True,
        )
        print(f"✓ Created ResNet50 model")

        trainable = model.get_trainable_params()
        frozen = model.get_frozen_params()
        total = trainable + frozen

        print(f"✓ Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"✓ Frozen params: {frozen:,} ({frozen/total*100:.1f}%)")

        # Test forward pass
        dummy_input = torch.rand(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"
        print(f"✓ Forward pass test passed: output shape={output.shape}")

        return True

    except Exception as e:
        print(f"ERROR: Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data preparation and setup")
    parser.add_argument(
        "--vindr_metadata",
        type=str,
        default=None,
        help="Path to VinDr-Mammo metadata CSV",
    )
    parser.add_argument(
        "--vindr_images",
        type=str,
        default=None,
        help="Path to VinDr-Mammo images directory",
    )
    parser.add_argument(
        "--inbreast_metadata",
        type=str,
        default=None,
        help="Path to INbreast metadata CSV",
    )
    parser.add_argument(
        "--inbreast_images",
        type=str,
        default=None,
        help="Path to INbreast images directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SETUP VERIFICATION TEST")
    print("=" * 60)

    # Check dependencies
    deps_ok = check_dependencies()

    # Test data loading
    loading_ok = test_data_loading()

    # Test model
    model_ok = test_model()

    # Check datasets if paths provided
    vindr_ok = True
    inbreast_ok = True

    if args.vindr_metadata:
        vindr_meta_ok = check_metadata(args.vindr_metadata, "VinDr-Mammo")
        if args.vindr_images:
            vindr_img_ok = check_images(args.vindr_metadata, args.vindr_images)
            vindr_ok = vindr_meta_ok and vindr_img_ok
        else:
            vindr_ok = vindr_meta_ok

    if args.inbreast_metadata:
        inbreast_meta_ok = check_metadata(args.inbreast_metadata, "INbreast")
        if args.inbreast_images:
            inbreast_img_ok = check_images(args.inbreast_metadata, args.inbreast_images)
            inbreast_ok = inbreast_meta_ok and inbreast_img_ok
        else:
            inbreast_ok = inbreast_meta_ok

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Dependencies: {'✓ PASS' if deps_ok else '✗ FAIL'}")
    print(f"Data loading: {'✓ PASS' if loading_ok else '✗ FAIL'}")
    print(f"Model: {'✓ PASS' if model_ok else '✗ FAIL'}")

    if args.vindr_metadata:
        print(f"VinDr-Mammo: {'✓ PASS' if vindr_ok else '✗ FAIL'}")
    if args.inbreast_metadata:
        print(f"INbreast: {'✓ PASS' if inbreast_ok else '✗ FAIL'}")

    all_ok = deps_ok and loading_ok and model_ok and vindr_ok and inbreast_ok

    if all_ok:
        print("\n✓ All tests passed! Setup is ready.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)
