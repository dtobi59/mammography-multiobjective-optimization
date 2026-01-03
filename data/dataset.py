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
