"""
Debug script to check INbreast data processing and identify evaluation issues.
"""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd
import numpy as np
from data.parsers import parse_dataset
import config

def debug_inbreast_parsing():
    """Debug INbreast metadata parsing."""
    print("=" * 80)
    print("DEBUGGING INBREAST PARSING")
    print("=" * 80)

    # Use local paths instead of config paths
    metadata_path = Path(__file__).parent / "INbreast.csv"
    image_dir = str(Path(__file__).parent / "images")

    print(f"\nMetadata path: {metadata_path}")
    print(f"Image directory: {image_dir}")

    try:
        # Parse using INbreastParser directly to avoid duplicate image_dir
        from data.parsers import INbreastParser
        parser = INbreastParser(
            metadata_path=str(metadata_path),
            image_dir=image_dir,
            metadata_format=config.INBREAST_CONFIG.get("metadata_format", "csv"),
            patient_id_col=config.INBREAST_CONFIG.get("patient_id_col"),
            laterality_col=config.INBREAST_CONFIG.get("laterality_col"),
            view_col=config.INBREAST_CONFIG.get("view_col"),
            birads_col=config.INBREAST_CONFIG.get("birads_col"),
            filename_col=config.INBREAST_CONFIG.get("filename_col"),
        )
        metadata = parser.parse()

        print(f"\n[OK] Successfully parsed {len(metadata)} images")
        print(f"\nColumns: {list(metadata.columns)}")
        print(f"\nLabel distribution:")
        print(metadata['label'].value_counts())

        print(f"\nBreast-level label distribution:")
        breast_labels = metadata.groupby('breast_id')['label'].max()
        print(breast_labels.value_counts())
        print(f"Total breasts: {len(breast_labels)}")
        print(f"Malignant breasts: {breast_labels.sum()}")
        print(f"Benign breasts: {len(breast_labels) - breast_labels.sum()}")

        print(f"\nSample of parsed data:")
        print(metadata[['image_id', 'patient_id', 'breast_id', 'view', 'label', 'birads_original']].head(10))

        # Check for unique image IDs
        print(f"\nUnique image IDs: {metadata['image_id'].nunique()}")
        print(f"Total rows: {len(metadata)}")
        if metadata['image_id'].nunique() != len(metadata):
            print("[WARNING] Duplicate image IDs found!")
            duplicates = metadata[metadata.duplicated(subset=['image_id'], keep=False)]
            print(f"Duplicates:\n{duplicates[['image_id', 'breast_id', 'label']]}")

        return metadata

    except Exception as e:
        print(f"\n[ERROR] Error parsing INbreast metadata: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_breast_aggregation(metadata):
    """Debug breast-level aggregation."""
    print("\n" + "=" * 80)
    print("DEBUGGING BREAST-LEVEL AGGREGATION")
    print("=" * 80)

    # Simulate predictions (random for debugging)
    np.random.seed(42)
    image_predictions = {
        img_id: np.random.rand() for img_id in metadata['image_id']
    }

    print(f"\nGenerated {len(image_predictions)} random predictions")
    print(f"Sample predictions: {list(image_predictions.items())[:5]}")

    # Test aggregation
    from utils.noisy_or import aggregate_to_breast_level

    breast_predictions, breast_labels = aggregate_to_breast_level(
        image_predictions, metadata
    )

    print(f"\nBreast-level results:")
    print(f"Total breasts: {len(breast_predictions)}")
    print(f"Predictions shape: {breast_predictions.shape}")
    print(f"Labels shape: {breast_labels.shape}")
    print(f"Predictions range: [{breast_predictions.min():.4f}, {breast_predictions.max():.4f}]")
    print(f"Label distribution: {np.bincount(breast_labels.astype(int))}")

    # Check for zeros
    zero_predictions = (breast_predictions == 0.0).sum()
    if zero_predictions > 0:
        print(f"\n[WARNING] {zero_predictions} breasts have prediction = 0.0")
        print("This suggests missing image predictions!")

    # Compute metrics
    from training.metrics import compute_metrics
    metrics = compute_metrics(breast_predictions, breast_labels)

    print(f"\nMetrics:")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Brier: {metrics['brier']:.4f}")

    # Check if only one class
    unique_labels = np.unique(breast_labels)
    if len(unique_labels) < 2:
        print(f"\n[CRITICAL] Only one class in labels: {unique_labels}")
        print("This explains the 0.0000 metrics!")

    return breast_predictions, breast_labels


def debug_image_id_matching():
    """Check if image IDs from dataset match metadata."""
    print("\n" + "=" * 80)
    print("DEBUGGING IMAGE ID MATCHING")
    print("=" * 80)

    # Use local path instead of config path
    metadata_path = Path(__file__).parent / "INbreast.csv"
    # Try comma first (updated format), fallback to semicolon
    try:
        df = pd.read_csv(metadata_path, sep=',')
    except:
        df = pd.read_csv(metadata_path, sep=';')

    print(f"\nRaw metadata shape: {df.shape}")
    print(f"\nRaw metadata columns: {list(df.columns)}")
    print(f"\nSample filenames from CSV:")
    if 'File Name' in df.columns:
        print(df['File Name'].head(10).tolist())

    # Check how image IDs are created
    image_ids_from_filename = df['File Name'].apply(lambda x: Path(str(x)).stem)
    print(f"\nImage IDs created from filename:")
    print(image_ids_from_filename.head(10).tolist())

    # Check for empty or malformed BI-RADS
    print(f"\nBI-RADS value counts:")
    print(df['Bi-Rads'].value_counts())

    # Check for rows with whitespace in BI-RADS
    birads_with_spaces = df[df['Bi-Rads'].astype(str).str.contains(r'\s{10,}')]
    if len(birads_with_spaces) > 0:
        print(f"\n[WARNING] {len(birads_with_spaces)} rows have excessive whitespace in BI-RADS:")
        print(birads_with_spaces[['Patient ID', 'File Name', 'Bi-Rads']])


if __name__ == "__main__":
    print("Starting INbreast evaluation debugging...")

    # Run all debug checks
    debug_image_id_matching()
    metadata = debug_inbreast_parsing()

    if metadata is not None:
        debug_breast_aggregation(metadata)

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
