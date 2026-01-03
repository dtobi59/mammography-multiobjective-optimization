"""
Zero-shot evaluation on target dataset (INbreast).
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
from typing import Dict
from models.resnet import ResNet50WithPartialFineTuning
from data.dataset import get_base_transform, MammographyDataset
from training.metrics import compute_metrics, compute_sensitivity_specificity
from utils.noisy_or import aggregate_to_breast_level
import config


def evaluate_target_zero_shot(
    model: torch.nn.Module,
    target_metadata: pd.DataFrame,
    image_dir: str,
    threshold: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Zero-shot evaluation on target dataset (INbreast).

    No fine-tuning, no threshold tuning - use transferred threshold from source.

    Args:
        model: Trained model (from source domain)
        target_metadata: Target dataset metadata
        image_dir: Directory containing target images
        threshold: Decision threshold (transferred from source)
        device: Device to evaluate on

    Returns:
        Dictionary of metrics: PR-AUC, AUROC, Brier, sensitivity, specificity
    """
    model = model.to(device)
    model.eval()

    # Create target dataloader (no augmentation, same preprocessing as source)
    base_transform = get_base_transform()
    target_dataset = MammographyDataset(
        metadata=target_metadata,
        image_dir=image_dir,
        transform=base_transform,
        augmentation=None,
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Collect image-level predictions
    image_predictions = {}

    with torch.no_grad():
        for images, labels, image_ids in target_loader:
            images = images.to(device)
            predictions = model(images).cpu().numpy()

            for img_id, pred in zip(image_ids, predictions):
                image_predictions[img_id] = float(pred)

    # Aggregate to breast-level using Noisy OR
    breast_predictions, breast_labels = aggregate_to_breast_level(
        image_predictions, target_metadata
    )

    # Compute metrics
    metrics = compute_metrics(breast_predictions, breast_labels)

    # Compute sensitivity and specificity at transferred threshold
    sensitivity, specificity = compute_sensitivity_specificity(
        breast_predictions, breast_labels, threshold
    )
    metrics["sensitivity"] = sensitivity
    metrics["specificity"] = specificity
    metrics["threshold"] = threshold

    return metrics


if __name__ == "__main__":
    """
    Example usage:

    python evaluation/evaluate_target.py --checkpoint path/to/checkpoint.pt --threshold 0.5 --hyperparameters config.json
    """
    import argparse
    from pathlib import Path
    from optimization.nsga3_runner import load_metadata

    parser = argparse.ArgumentParser(description="Zero-shot evaluation on INbreast (target dataset)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, required=True, help="Decision threshold from source")
    parser.add_argument(
        "--hyperparameters",
        type=str,
        required=True,
        help="Path to JSON file with hyperparameters",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Override default INbreast path")
    args = parser.parse_args()

    # Load INbreast metadata using dataset-specific parser
    dataset_path = args.dataset_path or config.INBREAST_PATH

    print("Loading INbreast metadata...")
    target_metadata = load_metadata(
        dataset_name="inbreast",
        dataset_path=dataset_path,
        dataset_config=config.INBREAST_CONFIG
    )

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
    image_dir = str(Path(dataset_path) / config.INBREAST_CONFIG["image_dir"])

    # Zero-shot evaluation
    print(f"Zero-shot evaluation on INbreast with threshold={args.threshold:.4f}")
    print("NOTE: No fine-tuning or threshold tuning on target data - pure zero-shot transfer")
    metrics = evaluate_target_zero_shot(
        model=model,
        target_metadata=target_metadata,
        image_dir=image_dir,
        threshold=args.threshold,
    )

    # Print results
    print("\n=== Zero-Shot INbreast Results ===")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Brier Score: {metrics['brier']:.4f}")
    print(f"Transferred Threshold: {args.threshold:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
