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


def compute_brier_null(labels: np.ndarray) -> float:
    """
    Compute null Brier score (predicting prevalence for all samples).

    The null Brier score is the Brier score achieved by always predicting
    the prevalence (mean of labels). This serves as a baseline for calibration.

    Formula:
        Brier_null = mean((prevalence - y_i)^2)
                   = prevalence * (1 - prevalence)

    Args:
        labels: Ground truth binary labels, shape (n_samples,)

    Returns:
        Null Brier score
    """
    prevalence = labels.mean()
    # Equivalent to: np.mean((prevalence - labels) ** 2)
    return prevalence * (1.0 - prevalence)


def compute_scaled_brier(brier: float, labels: np.ndarray) -> float:
    """
    Compute Scaled Brier score.

    Scaled Brier = 1 - (Brier / Brier_null)

    Range: (-∞, 1], where:
    - 1 = perfect calibration (Brier = 0)
    - 0 = no better than predicting prevalence
    - negative = worse than predicting prevalence

    Args:
        brier: Brier score from model predictions
        labels: Ground truth binary labels

    Returns:
        Scaled Brier score
    """
    brier_null = compute_brier_null(labels)

    # Handle edge case where all labels are the same
    if brier_null == 0:
        return 0.0

    return 1.0 - (brier / brier_null)


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
        - brier_null: Null Brier score (baseline)
        - scaled_brier: Scaled Brier score (1 - Brier/Brier_null)
    """
    # Handle edge cases
    if len(np.unique(labels)) < 2:
        # Only one class present, cannot compute PR-AUC and AUROC
        brier = brier_score_loss(labels, predictions)
        return {
            "pr_auc": 0.0,
            "auroc": 0.5,
            "brier": brier,
            "brier_null": 0.0,
            "scaled_brier": 0.0,
        }

    brier = brier_score_loss(labels, predictions)
    brier_null = compute_brier_null(labels)
    scaled_brier = compute_scaled_brier(brier, labels)

    metrics = {
        "pr_auc": average_precision_score(labels, predictions),
        "auroc": roc_auc_score(labels, predictions),
        "brier": brier,
        "brier_null": brier_null,
        "scaled_brier": scaled_brier,
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
