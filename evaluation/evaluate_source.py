"""
Evaluation on source validation set (VinDr-Mammo).
"""

# MUST BE FIRST - Fix imports
import sys
from pathlib import Path
_root = str(Path(__file__).parent.parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from models.resnet import ResNet50WithPartialFineTuning
from data.dataset import create_dataloaders, get_base_transform, MammographyDataset
from training.metrics import (
    compute_metrics,
    find_optimal_threshold,
    compute_sensitivity_specificity,
)
from training.robustness import RobustnessEvaluator
from utils.noisy_or import aggregate_to_breast_level
import config


def evaluate_source_validation(
    model: torch.nn.Module,
    val_metadata: pd.DataFrame,
    image_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Dict[str, float], float]:
    """
    Evaluate model on source validation set.

    Args:
        model: Trained model
        val_metadata: Validation metadata
        image_dir: Directory containing images
        device: Device to evaluate on

    Returns:
        Tuple of (metrics, optimal_threshold)
        - metrics: Dictionary with PR-AUC, AUROC, Brier, robustness, sensitivity, specificity
        - optimal_threshold: Optimal decision threshold
    """
    model = model.to(device)
    model.eval()

    # Create validation dataloader (no augmentation)
    base_transform = get_base_transform()
    val_dataset = MammographyDataset(
        metadata=val_metadata,
        image_dir=image_dir,
        transform=base_transform,
        augmentation=None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Collect image-level predictions
    image_predictions = {}

    with torch.no_grad():
        for images, labels, image_ids in val_loader:
            images = images.to(device)
            predictions = model(images).cpu().numpy()

            for img_id, pred in zip(image_ids, predictions):
                image_predictions[img_id] = float(pred)

    # Aggregate to breast-level
    breast_predictions, breast_labels = aggregate_to_breast_level(
        image_predictions, val_metadata
    )

    # Compute metrics
    metrics = compute_metrics(breast_predictions, breast_labels)

    # Compute robustness
    robustness_evaluator = RobustnessEvaluator(
        model=model,
        val_loader=val_loader,
        val_metadata=val_metadata,
        device=device,
    )
    robustness_degradation = robustness_evaluator.evaluate()
    metrics["robustness_degradation"] = robustness_degradation

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(breast_predictions, breast_labels)

    # Compute sensitivity and specificity at optimal threshold
    sensitivity, specificity = compute_sensitivity_specificity(
        breast_predictions, breast_labels, optimal_threshold
    )
    metrics["sensitivity"] = sensitivity
    metrics["specificity"] = specificity
    metrics["threshold"] = optimal_threshold

    return metrics, optimal_threshold


if __name__ == "__main__":
    """
    Example usage:

    python evaluation/evaluate_source.py --checkpoint path/to/checkpoint.pt --hyperparameters config.json
    """
    import argparse
    from pathlib import Path
    from data.dataset import create_train_val_split
    from optimization.nsga3_runner import load_metadata

    parser = argparse.ArgumentParser(description="Evaluate on source validation set (VinDr-Mammo)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--hyperparameters",
        type=str,
        required=True,
        help="Path to JSON file with hyperparameters",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Override default dataset path")
    args = parser.parse_args()

    # Load VinDr-Mammo metadata using dataset-specific parser
    dataset_path = args.dataset_path or config.VINDR_MAMMO_PATH

    print("Loading VinDr-Mammo metadata...")
    metadata = load_metadata(
        dataset_name="vindr",
        dataset_path=dataset_path,
        dataset_config=config.VINDR_CONFIG
    )
    _, val_metadata = create_train_val_split(metadata)

    # Load hyperparameters
    with open(args.hyperparameters, "r") as f:
        hparams = json.load(f)

    # Create model
    print("Creating model...")
    model = ResNet50WithPartialFineTuning(
        unfreeze_fraction=hparams["unfreeze_fraction"],
        dropout_rate=hparams["dropout_rate"],
        pretrained=False,  # Load from checkpoint
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get image directory
    image_dir = str(Path(dataset_path) / config.VINDR_CONFIG["image_dir"])

    # Evaluate
    print("Evaluating on source validation set...")
    metrics, threshold = evaluate_source_validation(
        model=model,
        val_metadata=val_metadata,
        image_dir=image_dir,
    )

    # Print results
    print("\n=== Source Validation Results ===")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Brier Score: {metrics['brier']:.4f}")
    print(f"Robustness Degradation: {metrics['robustness_degradation']:.4f}")
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
