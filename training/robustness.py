"""
Robustness evaluation under intensity perturbations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict
from data.augmentation import RobustnessPerturbation
from utils.noisy_or import aggregate_to_breast_level
from .metrics import compute_robustness_degradation
import config


class RobustnessEvaluator:
    """
    Evaluate model robustness under intensity perturbations.
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        val_metadata: pd.DataFrame,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize robustness evaluator.

        Args:
            model: Trained model
            val_loader: Validation dataloader (without augmentation)
            val_metadata: Validation metadata for breast-level aggregation
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.val_loader = val_loader
        self.val_metadata = val_metadata
        self.device = device
        self.perturbation = RobustnessPerturbation()

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate robustness degradation.

        Computes:
        - PR-AUC under standard inference
        - PR-AUC under perturbed inference
        - Robustness degradation R = PR-AUC_standard - PR-AUC_perturbed

        Returns:
            Robustness degradation (lower is better)
        """
        self.model.eval()

        # Collect predictions under standard and perturbed inference
        standard_predictions = {}
        perturbed_predictions = {}

        for images, labels, image_ids in self.val_loader:
            # Standard inference
            images_standard = images.to(self.device)
            preds_standard = self.model(images_standard).cpu().numpy()

            # Perturbed inference
            images_perturbed = torch.stack([
                self.perturbation(img, seed=config.RANDOM_SEED + i)
                for i, img in enumerate(images)
            ]).to(self.device)
            preds_perturbed = self.model(images_perturbed).cpu().numpy()

            # Store predictions
            for img_id, pred_std, pred_pert in zip(image_ids, preds_standard, preds_perturbed):
                standard_predictions[img_id] = float(pred_std)
                perturbed_predictions[img_id] = float(pred_pert)

        # Aggregate to breast-level
        breast_preds_standard, breast_labels = aggregate_to_breast_level(
            standard_predictions, self.val_metadata
        )
        breast_preds_perturbed, _ = aggregate_to_breast_level(
            perturbed_predictions, self.val_metadata
        )

        # Compute robustness degradation
        degradation = compute_robustness_degradation(
            breast_preds_standard,
            breast_preds_perturbed,
            breast_labels,
        )

        return degradation
