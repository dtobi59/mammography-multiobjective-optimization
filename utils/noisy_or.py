"""
Noisy OR aggregation for breast-level predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def noisy_or_aggregation(p_cc: float, p_mlo: float) -> float:
    """
    Compute breast-level probability using Noisy OR.

    Formula: p_breast = 1 - (1 - p_CC) * (1 - p_MLO)

    Args:
        p_cc: Probability from CC (craniocaudal) view
        p_mlo: Probability from MLO (mediolateral oblique) view

    Returns:
        Breast-level probability
    """
    return 1.0 - (1.0 - p_cc) * (1.0 - p_mlo)


def aggregate_to_breast_level(
    image_predictions: Dict[str, float],
    metadata: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate image-level predictions to breast-level using generalized Noisy OR.

    Generalized Noisy OR formula for m images:
        p_breast = 1 - ∏_{j=1}^{m} (1 - p_j)

    Special cases:
        - m = 1: p_breast = p_1 (single view)
        - m = 2: p_breast = 1 - (1 - p_1)(1 - p_2) (standard Noisy OR)
        - m > 2: generalized formula applies

    Args:
        image_predictions: Dictionary mapping image_id to predicted probability
        metadata: DataFrame with columns: image_id, patient_id, breast_id, view, label
                  breast_id uniquely identifies each breast
                  label is the breast-level ground truth

    Returns:
        Tuple of (breast_predictions, breast_labels)
        - breast_predictions: numpy array of breast-level probabilities
        - breast_labels: numpy array of breast-level ground truth labels
    """
    # Group by breast_id
    breast_groups = metadata.groupby("breast_id")

    breast_predictions = []
    breast_labels = []

    for breast_id, group in breast_groups:
        # Collect all available image predictions for this breast
        image_probs = []
        for img_id in group["image_id"]:
            if img_id in image_predictions:
                image_probs.append(image_predictions[img_id])

        # Apply generalized Noisy OR
        if len(image_probs) == 0:
            # No predictions available, default to 0
            breast_prob = 0.0
        elif len(image_probs) == 1:
            # Single view: p_breast = p_1
            breast_prob = image_probs[0]
        else:
            # Multiple views: p_breast = 1 - ∏(1 - p_j)
            breast_prob = 1.0 - np.prod([1.0 - p for p in image_probs])

        breast_predictions.append(breast_prob)

        # Ground truth label: max across all views (if ANY view is positive, breast is positive)
        # This follows clinical logic and handles cases where views may have different assessments
        breast_label = int(group["label"].max())
        breast_labels.append(breast_label)

    return np.array(breast_predictions), np.array(breast_labels)
