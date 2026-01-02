"""
Training pipeline with early stopping.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .metrics import compute_metrics
from utils.noisy_or import aggregate_to_breast_level
import config


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        mode: str = "max",
        delta: float = 0.0,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            mode: 'max' for metrics to maximize (e.g., PR-AUC), 'min' for metrics to minimize
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            current_score: Current validation metric value
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        # Check for improvement
        if self.mode == "max":
            improved = current_score > self.best_score + self.delta
        else:
            improved = current_score < self.best_score - self.delta

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Trainer for breast cancer classification with early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_metadata: pd.DataFrame,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int = config.MAX_EPOCHS,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            val_metadata: Validation metadata for breast-level aggregation
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_epochs: Maximum number of training epochs
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_metadata = val_metadata
        self.max_epochs = max_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            mode="max",  # Maximize PR-AUC
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_pr_auc": [],
            "val_auroc": [],
            "val_brier": [],
        }

        self.best_checkpoint_path = None

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            # Forward pass
            predictions = self.model(images)
            loss = self.criterion(predictions, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set with breast-level aggregation.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        # Collect image-level predictions
        image_predictions = {}
        image_labels = {}

        for images, labels, image_ids in self.val_loader:
            images = images.to(self.device)
            predictions = self.model(images).cpu().numpy()

            for img_id, pred, label in zip(image_ids, predictions, labels.numpy()):
                image_predictions[img_id] = float(pred)
                image_labels[img_id] = int(label)

        # Aggregate to breast-level using Noisy OR
        breast_predictions, breast_labels = aggregate_to_breast_level(
            image_predictions, self.val_metadata
        )

        # Compute metrics
        metrics = compute_metrics(breast_predictions, breast_labels)

        return metrics

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def train(self) -> Dict[str, float]:
        """
        Train model with early stopping.

        Returns:
            Best validation metrics
        """
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # Validate
            val_metrics = self.validate()
            self.history["val_pr_auc"].append(val_metrics["pr_auc"])
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["val_brier"].append(val_metrics["brier"])

            # Print progress
            print(f"Epoch {epoch + 1}/{self.max_epochs} - "
                  f"Loss: {train_loss:.4f}, "
                  f"Val PR-AUC: {val_metrics['pr_auc']:.4f}, "
                  f"Val AUROC: {val_metrics['auroc']:.4f}, "
                  f"Val Brier: {val_metrics['brier']:.4f}")

            # Save checkpoint if best so far
            val_pr_auc = val_metrics["pr_auc"]
            if self.early_stopping.best_score is None or val_pr_auc > self.early_stopping.best_score:
                if self.checkpoint_dir is not None:
                    self.best_checkpoint_path = os.path.join(
                        self.checkpoint_dir, f"best_checkpoint.pt"
                    )
                    self.save_checkpoint(self.best_checkpoint_path)

            # Check early stopping
            if self.early_stopping(val_pr_auc, epoch):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Restore best checkpoint
        if self.best_checkpoint_path is not None and os.path.exists(self.best_checkpoint_path):
            print(f"Restoring best checkpoint from epoch {self.early_stopping.best_epoch + 1}")
            self.load_checkpoint(self.best_checkpoint_path)

        # Return best validation metrics
        best_epoch = self.early_stopping.best_epoch
        best_metrics = {
            "pr_auc": self.history["val_pr_auc"][best_epoch],
            "auroc": self.history["val_auroc"][best_epoch],
            "brier": self.history["val_brier"][best_epoch],
        }

        return best_metrics
