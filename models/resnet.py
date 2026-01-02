"""
ResNet-50 with partial fine-tuning and dropout.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional


class ResNet50WithPartialFineTuning(nn.Module):
    """
    ResNet-50 with ImageNet pretrained weights and partial fine-tuning.

    The backbone can be partially frozen/unfrozen based on unfreeze_fraction parameter.
    Binary classification head with dropout and sigmoid output.
    """

    def __init__(
        self,
        unfreeze_fraction: float = 1.0,
        dropout_rate: float = 0.0,
        pretrained: bool = True,
    ):
        """
        Initialize ResNet-50 model.

        Args:
            unfreeze_fraction: Fraction of backbone layers to unfreeze, in [0, 1]
                               0.0 = freeze all backbone (only train classifier)
                               1.0 = unfreeze all backbone (full fine-tuning)
            dropout_rate: Dropout rate before final classification layer
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        self.unfreeze_fraction = max(0.0, min(1.0, unfreeze_fraction))
        self.dropout_rate = dropout_rate

        # Load pretrained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = resnet50(weights=weights)

        # Remove original classification head
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Get feature dimension (2048 for ResNet-50)
        feature_dim = backbone.fc.in_features

        # Binary classification head with dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(feature_dim, 1)

        # Apply partial fine-tuning
        self._setup_partial_finetuning()

    def _setup_partial_finetuning(self) -> None:
        """
        Freeze/unfreeze backbone layers based on unfreeze_fraction.

        Strategy: Unfreeze layers from the end (deeper layers) backward.
        - unfreeze_fraction = 0.0: Freeze all backbone layers
        - unfreeze_fraction = 1.0: Unfreeze all backbone layers
        - unfreeze_fraction = 0.5: Unfreeze last 50% of layers
        """
        # Get all layers in the backbone
        all_layers = list(self.features.children())
        n_layers = len(all_layers)

        # Calculate number of layers to unfreeze (from the end)
        n_unfreeze = int(n_layers * self.unfreeze_fraction)

        # Freeze all layers first
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze the last n_unfreeze layers
        if n_unfreeze > 0:
            layers_to_unfreeze = all_layers[-n_unfreeze:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

        # Classification head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, 3, H, W)

        Returns:
            Output probabilities, shape (batch_size,)
        """
        # Extract features
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        # Classification
        features = self.dropout(features)
        logits = self.classifier(features)

        # Sigmoid for binary classification
        probs = torch.sigmoid(logits).squeeze(1)  # (batch_size,)

        return probs

    def get_trainable_params(self) -> int:
        """
        Get number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_params(self) -> int:
        """
        Get number of frozen parameters.

        Returns:
            Number of frozen parameters
        """
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
