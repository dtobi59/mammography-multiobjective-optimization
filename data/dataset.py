"""
Mammography dataset with patient-wise train/validation split.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable
from pathlib import Path
from datetime import datetime
import config


def create_train_val_split(
    metadata: pd.DataFrame,
    train_ratio: float = config.TRAIN_VAL_SPLIT,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create patient-wise train/validation split.

    Args:
        metadata: DataFrame with at least columns: patient_id, image_id
        train_ratio: Fraction of patients for training (default 0.8)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_metadata, val_metadata)
    """
    # Get unique patient IDs
    patient_ids = metadata["patient_id"].unique()

    # Shuffle patients with fixed seed
    rng = np.random.RandomState(random_seed)
    shuffled_patients = rng.permutation(patient_ids)

    # Split patients
    n_train = int(len(shuffled_patients) * train_ratio)
    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train:])

    # Split metadata by patient
    train_metadata = metadata[metadata["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_metadata = metadata[metadata["patient_id"].isin(val_patients)].reset_index(drop=True)

    return train_metadata, val_metadata


def create_stratified_subsample(
    metadata: pd.DataFrame,
    target_total: int = 1000,
    target_malignant: int = 250,
    target_benign: int = 750,
    train_ratio: float = config.TRAIN_VAL_SPLIT,
    random_seed: int = config.RANDOM_SEED,
    output_dir: str = "./manifests",
    exclude_birads_3: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Create stratified subsample with patient-wise split and save manifest.

    IMPORTANT: Patient-wise split happens FIRST, then stratified sampling within
    each partition to prevent data leakage.

    Args:
        metadata: Full VinDr-Mammo metadata with columns:
                  image_id, patient_id, breast_id, view, label, image_path, birads_original
        target_total: Total target images (default: 1000)
        target_malignant: Target malignant images (default: 250)
        target_benign: Target benign images (default: 750)
        train_ratio: Train/val split ratio (default: 0.8)
        random_seed: Random seed for reproducibility
        output_dir: Directory to save manifest
        exclude_birads_3: Whether to exclude BI-RADS 3 (default: True)

    Returns:
        Tuple of (train_metadata, val_metadata, manifest_path)
    """
    rng = np.random.RandomState(random_seed)

    # Step 1: Exclude BI-RADS 3 if requested
    if exclude_birads_3:
        # Check if birads_original column exists, otherwise infer from label
        if 'birads_original' in metadata.columns:
            # Exclude rows where birads_original is "3" or contains "3"
            metadata = metadata[~metadata['birads_original'].astype(str).str.contains('3', na=False)].copy()
        else:
            # If no birads_original column, we can't exclude BI-RADS 3 explicitly
            print("Warning: birads_original column not found, cannot exclude BI-RADS 3")

    print(f"After excluding BI-RADS 3: {len(metadata)} images")

    # Step 2: Patient-wise split (80/20) to prevent data leakage
    train_meta_full, val_meta_full = create_train_val_split(
        metadata, train_ratio=train_ratio, random_seed=random_seed
    )

    # Step 3: Calculate target counts for train and val
    target_train_malignant = int(target_malignant * train_ratio)
    target_train_benign = int(target_benign * train_ratio)
    target_val_malignant = target_malignant - target_train_malignant
    target_val_benign = target_benign - target_train_benign

    # Step 4: Stratified sampling within each partition
    def stratified_sample(meta: pd.DataFrame, n_malignant: int, n_benign: int, rng: np.random.RandomState) -> pd.DataFrame:
        """Sample n_malignant malignant and n_benign benign images."""
        malignant = meta[meta['label'] == 1]
        benign = meta[meta['label'] == 0]

        # Sample with replacement if not enough samples
        replace_malignant = len(malignant) < n_malignant
        replace_benign = len(benign) < n_benign

        if replace_malignant:
            print(f"  Warning: Only {len(malignant)} malignant images available, need {n_malignant}. Adjusting...")
            n_malignant = len(malignant)

        if replace_benign:
            print(f"  Warning: Only {len(benign)} benign images available, need {n_benign}. Adjusting...")
            n_benign = len(benign)

        malignant_sample = malignant.sample(n=n_malignant, replace=False, random_state=rng)
        benign_sample = benign.sample(n=n_benign, replace=False, random_state=rng)

        return pd.concat([malignant_sample, benign_sample], ignore_index=True)

    print(f"\nTargeting {target_total} total images:")
    print(f"  Train: {target_train_malignant + target_train_benign} images "
          f"({target_train_malignant} malignant, {target_train_benign} benign)")
    print(f"  Val: {target_val_malignant + target_val_benign} images "
          f"({target_val_malignant} malignant, {target_val_benign} benign)")

    # Sample from train partition
    print("\nSampling from train partition...")
    train_metadata = stratified_sample(train_meta_full, target_train_malignant, target_train_benign, rng)
    train_metadata = train_metadata.copy()
    train_metadata['split'] = 'train'

    # Sample from val partition
    print("Sampling from val partition...")
    val_metadata = stratified_sample(val_meta_full, target_val_malignant, target_val_benign, rng)
    val_metadata = val_metadata.copy()
    val_metadata['split'] = 'val'

    # Step 5: Create and save manifest
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = os.path.join(output_dir, f"vindr_subsample_seed{random_seed}_{timestamp}.csv")

    manifest = pd.concat([train_metadata, val_metadata], ignore_index=True)

    # Ensure manifest has required columns
    required_cols = ['image_id', 'patient_id', 'breast_id', 'laterality', 'view', 'label', 'image_path', 'split']
    if 'birads_original' in manifest.columns:
        required_cols.append('birads_original')

    # Select only required columns (if they exist)
    manifest_cols = [col for col in required_cols if col in manifest.columns]
    manifest = manifest[manifest_cols]

    manifest.to_csv(manifest_path, index=False)

    # Step 6: Log final counts
    print("\n" + "=" * 80)
    print("DATASET SUBSAMPLING SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(manifest)}")
    print(f"  Malignant: {(manifest['label'] == 1).sum()}")
    print(f"  Benign: {(manifest['label'] == 0).sum()}")
    print()
    print(f"Train: {len(train_metadata)} images, {train_metadata['breast_id'].nunique()} breasts")
    print(f"  Malignant: {(train_metadata['label'] == 1).sum()}")
    print(f"  Benign: {(train_metadata['label'] == 0).sum()}")
    print()
    print(f"Val: {len(val_metadata)} images, {val_metadata['breast_id'].nunique()} breasts")
    print(f"  Malignant: {(val_metadata['label'] == 1).sum()}")
    print(f"  Benign: {(val_metadata['label'] == 0).sum()}")
    print()
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 80)

    # Drop the 'split' column from return values (keep in manifest only)
    train_metadata = train_metadata.drop(columns=['split'])
    val_metadata = val_metadata.drop(columns=['split'])

    return train_metadata, val_metadata, manifest_path


class MammographyDataset(Dataset):
    """
    Dataset for mammography images.

    Expects:
    - Grayscale PNG images
    - Metadata CSV with columns: image_id, patient_id, breast_id, view, label, image_path
    - Image-level training, breast-level evaluation
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            metadata: DataFrame with image metadata
            image_dir: Directory containing PNG images
            transform: Base transforms (e.g., resize, normalization)
            augmentation: Augmentation pipeline (applied after transform, only during training)
        """
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get image, label, and image_id.

        Args:
            idx: Index

        Returns:
            Tuple of (image, label, image_id)
            - image: Tensor of shape (C, H, W)
            - label: Binary label (0 or 1)
            - image_id: String identifier for the image
        """
        row = self.metadata.iloc[idx]

        # Load image
        image_path = os.path.join(self.image_dir, row["image_path"])
        image = Image.open(image_path).convert("L")  # Grayscale

        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(np.array(image)).float() / 255.0

        # Add channel dimension: (H, W) -> (1,H, W)
        image = image.unsqueeze(0)

        # Apply base transform (resize, etc.)
        if self.transform is not None:
            image = self.transform(image)

        # Convert to 3-channel for ResNet (repeat grayscale across channels)
        image = image.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)

        # Apply augmentation (training only)
        if self.augmentation is not None:
            image = self.augmentation(image)

        # Get label and image_id
        label = int(row["label"])
        image_id = row["image_id"]

        return image, label, image_id


def get_base_transform(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> Callable:
    """
    Get base transform for resizing images.

    Args:
        image_size: Target image size (height, width)

    Returns:
        Transform function
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(image_size),
    ])


def create_dataloaders(
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    image_dir: str,
    batch_size: int = config.BATCH_SIZE,
    augmentation_strength: float = 0.0,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_metadata: Training metadata
        val_metadata: Validation metadata
        image_dir: Directory containing images
        batch_size: Batch size
        augmentation_strength: Augmentation strength for training
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from .augmentation import IntensityAugmentation

    base_transform = get_base_transform()

    # Training dataset with augmentation
    train_augmentation = IntensityAugmentation(strength=augmentation_strength)
    train_dataset = MammographyDataset(
        metadata=train_metadata,
        image_dir=image_dir,
        transform=base_transform,
        augmentation=train_augmentation,
    )

    # Validation dataset without augmentation
    val_dataset = MammographyDataset(
        metadata=val_metadata,
        image_dir=image_dir,
        transform=base_transform,
        augmentation=None,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
