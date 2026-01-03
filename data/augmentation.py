"""
Intensity-based augmentation with strength parameter.
"""

import torch
import torchvision.transforms.functional as TF
from typing import Optional
import config


class IntensityAugmentation:
    """
    Intensity-only augmentation with scalar strength parameter.

    Augmentations include:
    - Brightness adjustment
    - Contrast scaling
    - Additive Gaussian noise

    No geometric transforms are applied.
    """

    def __init__(
        self,
        strength: float = 0.0,
        brightness_factor: float = config.AUGMENTATION_CONFIG["brightness_factor"],
        contrast_factor: float = config.AUGMENTATION_CONFIG["contrast_factor"],
        noise_std: float = config.AUGMENTATION_CONFIG["noise_std"],
    ):
        """
        Initialize augmentation pipeline.

        Args:
            strength: Augmentation strength in [0, 1]. Linearly scales magnitude of all augmentations.
            brightness_factor: Maximum brightness adjustment (scaled by strength)
            contrast_factor: Maximum contrast scaling (scaled by strength)
            noise_std: Standard deviation of Gaussian noise (scaled by strength)
        """
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply intensity augmentations to image.

        Args:
            image: Input image tensor, shape (C, H, W), values in [0, 1]

        Returns:
            Augmented image tensor, shape (C, H, W), values clipped to [0, 1]
        """
        if self.strength == 0.0:
            return image

        # Random brightness adjustment: multiply by factor in [1-delta, 1+delta]
        brightness_delta = self.brightness_factor * self.strength
        brightness_mult = 1.0 + torch.empty(1).uniform_(-brightness_delta, brightness_delta).item()
        image = TF.adjust_brightness(image, brightness_mult)

        # Random contrast scaling: multiply by factor in [1-delta, 1+delta]
        contrast_delta = self.contrast_factor * self.strength
        contrast_mult = 1.0 + torch.empty(1).uniform_(-contrast_delta, contrast_delta).item()
        image = TF.adjust_contrast(image, contrast_mult)

        # Additive Gaussian noise
        noise_std_scaled = self.noise_std * self.strength
        if noise_std_scaled > 0:
            noise = torch.randn_like(image) * noise_std_scaled
            image = image + noise

        # Clip to valid range [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image


class RobustnessPerturbation:
    """
    Mild intensity perturbations for robustness evaluation.

    Fixed perturbations (not random) for consistent robustness testing.
    """

    def __init__(
        self,
        brightness_delta: float = config.ROBUSTNESS_PERTURBATION["brightness_delta"],
        contrast_delta: float = config.ROBUSTNESS_PERTURBATION["contrast_delta"],
        noise_std: float = config.ROBUSTNESS_PERTURBATION["noise_std"],
    ):
        """
        Initialize robustness perturbation.

        Args:
            brightness_delta: Fixed brightness adjustment
            contrast_delta: Fixed contrast scaling
            noise_std: Standard deviation of Gaussian noise
        """
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Apply fixed perturbation to image.

        Args:
            image: Input image tensor, shape (C, H, W), values in [0, 1]
            seed: Optional seed for noise reproducibility

        Returns:
            Perturbed image tensor, shape (C, H, W), values clipped to [0, 1]
        """
        # Apply fixed brightness and contrast adjustments
        image = TF.adjust_brightness(image, 1.0 + self.brightness_delta)
        image = TF.adjust_contrast(image, 1.0 + self.contrast_delta)

        # Add Gaussian noise with optional seed
        if self.noise_std > 0:
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
                noise = torch.randn(image.shape, generator=generator, dtype=image.dtype, device=image.device) * self.noise_std
            else:
                noise = torch.randn_like(image) * self.noise_std
            image = image + noise

        # Clip to valid range [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image
