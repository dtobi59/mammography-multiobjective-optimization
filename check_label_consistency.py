"""Check label consistency with actual patient IDs."""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd
import config
from data.parsers import INbreastParser

# Parse using actual patient IDs
metadata_path = Path(__file__).parent / "INbreast.csv"
image_dir = str(Path(__file__).parent / "images")

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

print("=" * 80)
print("LABEL CONSISTENCY WITH ACTUAL PATIENT IDs")
print("=" * 80)

# Check label consistency within breasts
inconsistent = 0
total_multi_image = 0
inconsistent_breasts = []

for breast_id, group in metadata.groupby("breast_id"):
    if len(group) > 1:
        total_multi_image += 1
        labels = group['label'].values
        if len(set(labels)) > 1:
            inconsistent += 1
            inconsistent_breasts.append({
                'breast_id': breast_id,
                'patient_id': group['patient_id'].iloc[0],
                'num_images': len(group),
                'labels': labels.tolist(),
                'birads': group['birads_original'].tolist(),
                'views': group['view'].tolist()
            })

print(f"\nTotal breasts: {metadata['breast_id'].nunique()}")
print(f"Breasts with >1 image: {total_multi_image}")
print(f"Breasts with inconsistent labels: {inconsistent} ({inconsistent/total_multi_image*100:.1f}%)")

# Check breast-level label distribution
breast_labels = metadata.groupby('breast_id')['label'].max()
print(f"\nBreast-level labels: Benign={len(breast_labels[breast_labels==0])}, Malignant={len(breast_labels[breast_labels==1])}")

# Show some inconsistent examples
if inconsistent > 0:
    print(f"\n{'-'*80}")
    print("Sample Inconsistent Breasts (first 10):")
    print('-'*80)
    for i, breast in enumerate(inconsistent_breasts[:10]):
        print(f"\nBreast {i+1}: {breast['breast_id']}")
        print(f"  Patient: {breast['patient_id']}")
        print(f"  Views: {breast['views']}")
        print(f"  Labels: {breast['labels']}")
        print(f"  BI-RADS: {breast['birads']}")

# Patient-level statistics
print(f"\n{'-'*80}")
print("PATIENT-LEVEL STATISTICS")
print('-'*80)
print(f"Total patients: {metadata['patient_id'].nunique()}")
print(f"Total breasts: {metadata['breast_id'].nunique()}")
print(f"Average breasts per patient: {metadata['breast_id'].nunique() / metadata['patient_id'].nunique():.2f}")

# Images per breast distribution
images_per_breast = metadata.groupby('breast_id').size()
print(f"\nImages per breast distribution:")
print(images_per_breast.value_counts().sort_index())
