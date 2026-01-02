"""
Metric computation for breast cancer classification.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from typing import Dict, Tuple
import config


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted probabilities, shape (n_samples,)
        labels: Ground truth binary labels, shape (n_samples,)

    Returns:
        Dictionary of metrics:
        - pr_auc: Precision-Recall AUC
        - auroc: ROC AUC
        - brier: Brier score
    """
    # Handle edge cases
    if len(np.unique(labels)) < 2:
        # Only one class present, cannot compute PR-AUC and AUROC
        return {
            "pr_auc": 0.0,
            "auroc": 0.5,
            "brier": brier_score_loss(labels, predictions),
        }

    metrics = {
        "pr_auc": average_precision_score(labels, predictions),
        "auroc": roc_auc_score(labels, predictions),
        "brier": brier_score_loss(labels, predictions),
    }

    return metrics


def compute_robustness_degradation(
    predictions_standard: np.ndarray,
    predictions_perturbed: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute robustness degradation.

    Robustness degradation R = PR-AUC_standard - PR-AUC_perturbed

    Args:
        predictions_standard: Predictions under standard inference
        predictions_perturbed: Predictions under perturbed inference
        labels: Ground truth labels

    Returns:
        Robustness degradation (lower is better, can be negative)
    """
    # Handle edge cases
    if len(np.unique(labels)) < 2:
        return 0.0

    pr_auc_standard = average_precision_score(labels, predictions_standard)
    pr_auc_perturbed = average_precision_score(labels, predictions_perturbed)

    degradation = pr_auc_standard - pr_auc_perturbed

    return degradation


def find_optimal_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    method: str = config.THRESHOLD_SELECTION_METHOD,
) -> float:
    """
    Find optimal decision threshold.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        method: Threshold selection method ('youden_j' for Youden's J statistic)

    Returns:
        Optimal threshold
    """
    if method == "youden_j":
        # Youden's J statistic: sensitivity + specificity - 1
        # Find threshold that maximizes J
        thresholds = np.linspace(0, 1, 101)
        best_j = -1
        best_threshold = 0.5

        for threshold in thresholds:
            binary_preds = (predictions >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()

            # Avoid division by zero
            if (tp + fn) > 0 and (tn + fp) > 0:
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                j = sensitivity + specificity - 1

                if j > best_j:
                    best_j = j
                    best_threshold = threshold

        return best_threshold
    else:
        # Default: 0.5
        return 0.5


def compute_sensitivity_specificity(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """
    Compute sensitivity and specificity at a given threshold.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        threshold: Decision threshold

    Returns:
        Tuple of (sensitivity, specificity)
    """
    binary_preds = (predictions >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()

    # Avoid division by zero
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity
