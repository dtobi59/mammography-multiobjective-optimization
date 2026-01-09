"""
Verify breast-level label consistency in datasets.

This script checks if all images from the same breast have consistent labels,
which is an assumption made by the current breast-level aggregation code.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from optimization.nsga3_runner import load_metadata


def check_label_consistency(metadata: pd.DataFrame, dataset_name: str):
    """
    Check if all images from the same breast have consistent labels.

    Args:
        metadata: DataFrame with breast_id and label columns
        dataset_name: Name of dataset for reporting
    """
    print("=" * 80)
    print(f"BREAST-LEVEL LABEL CONSISTENCY CHECK: {dataset_name}")
    print("=" * 80)
    print()

    # Group by breast_id
    breast_groups = metadata.groupby("breast_id")

    inconsistent_breasts = []
    total_breasts = 0

    for breast_id, group in breast_groups:
        total_breasts += 1
        labels = group["label"].unique()

        if len(labels) > 1:
            # Found inconsistent labels for same breast
            inconsistent_breasts.append({
                "breast_id": breast_id,
                "labels": labels.tolist(),
                "images": group["image_id"].tolist(),
                "views": group["view"].tolist() if "view" in group.columns else None
            })

    # Report results
    print(f"Total breasts: {total_breasts}")
    print(f"Total images: {len(metadata)}")
    print(f"Images per breast: {len(metadata) / total_breasts:.2f} (avg)")
    print()

    if len(inconsistent_breasts) == 0:
        print("[OK] ALL BREASTS HAVE CONSISTENT LABELS")
        print("  All images from the same breast have the same label.")
        print("  Using .iloc[0] is SAFE for this dataset.")
    else:
        print(f"[!] FOUND {len(inconsistent_breasts)} BREASTS WITH INCONSISTENT LABELS")
        print(f"  {len(inconsistent_breasts) / total_breasts * 100:.1f}% of breasts have mixed labels")
        print()
        print("  This means some breasts have BOTH malignant AND benign views,")
        print("  and using .iloc[0] could give WRONG breast-level labels!")
        print()
        print("  Examples of inconsistent breasts:")

        for i, breast in enumerate(inconsistent_breasts[:5]):  # Show first 5
            print(f"\n  Breast {i+1}: {breast['breast_id']}")
            print(f"    Labels found: {breast['labels']}")
            print(f"    Images: {breast['images']}")
            if breast['views']:
                print(f"    Views: {breast['views']}")

    print()
    print("=" * 80)
    print()

    return len(inconsistent_breasts) > 0


def recommend_fix():
    """Print recommendation for fixing the issue."""
    print("=" * 80)
    print("RECOMMENDED FIX")
    print("=" * 80)
    print()
    print("In utils/noisy_or.py, line 79, change:")
    print()
    print("  [X] CURRENT (INCORRECT):")
    print("     breast_label = group['label'].iloc[0]")
    print()
    print("  [OK] RECOMMENDED (CORRECT):")
    print("     breast_label = group['label'].max()")
    print()
    print("Rationale:")
    print("  - If ANY view shows malignancy (label=1), the breast is malignant")
    print("  - Follows clinical logic: positive in any view -> positive breast")
    print("  - max() handles both consistent and inconsistent cases correctly")
    print("=" * 80)
    print()


if __name__ == "__main__":
    vindr_has_issue = False
    inbreast_has_issue = False

    # Check VinDr-Mammo (if available)
    if hasattr(config, 'VINDR_PATH'):
        try:
            print("\n" + "=" * 80)
            print("LOADING VINDR-MAMMO DATASET")
            print("=" * 80)

            vindr_metadata = load_metadata(
                dataset_name="vindr",
                dataset_path=config.VINDR_PATH,
                dataset_config=config.VINDR_CONFIG
            )
            print(f"Loaded {len(vindr_metadata)} images")
            print()

            vindr_has_issue = check_label_consistency(vindr_metadata, "VinDr-Mammo")
        except Exception as e:
            print(f"[!] Could not load VinDr-Mammo: {e}")
            print()
    else:
        print("\n[!] VINDR_PATH not configured, skipping VinDr-Mammo check")
        print()

    # Check INbreast (if available)
    if hasattr(config, 'INBREAST_PATH'):
        try:
            print("\n" + "=" * 80)
            print("LOADING INBREAST DATASET")
            print("=" * 80)

            inbreast_metadata = load_metadata(
                dataset_name="inbreast",
                dataset_path=config.INBREAST_PATH,
                dataset_config=config.INBREAST_CONFIG
            )
            print(f"Loaded {len(inbreast_metadata)} images")
            print()

            inbreast_has_issue = check_label_consistency(inbreast_metadata, "INbreast")
        except Exception as e:
            print(f"[!] Could not load INbreast: {e}")
            print()
    else:
        print("\n[!] INBREAST_PATH not configured, skipping INbreast check")
        print()

    # Provide recommendation if issues found
    if vindr_has_issue or inbreast_has_issue:
        print()
        recommend_fix()
    else:
        print()
        print("=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        print()
        print("[OK] No label inconsistencies found in available datasets.")
        print("  Current implementation (.iloc[0]) is safe for these datasets.")
        print()
        print("However, consider using .max() for robustness:")
        print("  - Handles edge cases if they exist in other datasets")
        print("  - Follows clinical best practice")
        print("  - No performance impact")
        print("=" * 80)
