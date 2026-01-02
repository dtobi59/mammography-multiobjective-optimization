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
    Aggregate image-level predictions to breast-level using Noisy OR.

    Args:
        image_predictions: Dictionary mapping image_id to predicted probability
        metadata: DataFrame with columns: image_id, patient_id, breast_id, view, label
                  view should be 'CC' or 'MLO'
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
        # Get predictions for CC and MLO views
        cc_views = group[group["view"] == "CC"]
        mlo_views = group[group["view"] == "MLO"]

        # Handle cases where views might be missing (use 0 probability)
        if len(cc_views) == 0:
            p_cc = 0.0
        else:
            # If multiple CC views, take max probability (most suspicious)
            p_cc = max([image_predictions.get(img_id, 0.0) for img_id in cc_views["image_id"]])

        if len(mlo_views) == 0:
            p_mlo = 0.0
        else:
            # If multiple MLO views, take max probability (most suspicious)
            p_mlo = max([image_predictions.get(img_id, 0.0) for img_id in mlo_views["image_id"]])

        # Apply Noisy OR
        breast_prob = noisy_or_aggregation(p_cc, p_mlo)
        breast_predictions.append(breast_prob)

        # Ground truth label (same for all views of the same breast)
        breast_label = group["label"].iloc[0]
        breast_labels.append(breast_label)

    return np.array(breast_predictions), np.array(breast_labels)
