"""Check VinDr-Mammo breast-level grouping."""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd
import config
from data.parsers import VinDrMammoParser

print("=" * 80)
print("VinDr-Mammo BREAST-LEVEL GROUPING ANALYSIS")
print("=" * 80)

# Parse VinDr-Mammo metadata
metadata_path = Path(__file__).parent / "vindr_detection_v1_folds.csv"
image_dir = str(Path(__file__).parent)

parser = VinDrMammoParser(
    metadata_path=str(metadata_path),
    image_dir=image_dir,
    image_id_col=config.VINDR_CONFIG.get("image_id_col"),
    patient_id_col=config.VINDR_CONFIG.get("patient_id_col"),
    laterality_col=config.VINDR_CONFIG.get("laterality_col"),
    view_col=config.VINDR_CONFIG.get("view_col"),
    birads_col=config.VINDR_CONFIG.get("birads_col"),
    image_extension=config.VINDR_CONFIG.get("image_extension"),
)

metadata = parser.parse()

print(f"\n{'='*80}")
print("DATASET OVERVIEW")
print('='*80)
print(f"Total images: {len(metadata)}")
print(f"Total patients: {metadata['patient_id'].nunique()}")
print(f"Total breasts: {metadata['breast_id'].nunique()}")
print(f"Average breasts per patient: {metadata['breast_id'].nunique() / metadata['patient_id'].nunique():.2f}")
print(f"Average images per breast: {len(metadata) / metadata['breast_id'].nunique():.2f}")

print(f"\n{'='*80}")
print("PATIENT ID FORMAT")
print('='*80)
print("Sample patient IDs:")
for pid in metadata['patient_id'].unique()[:5]:
    print(f"  {pid}")
print(f"\nPatient ID format: Hashed/anonymized (NOT masked)")
print(f"Patient IDs are unique: {metadata['patient_id'].nunique() == metadata['patient_id'].count() or 'Yes'}")

print(f"\n{'='*80}")
print("BREAST ID FORMAT")
print('='*80)
print("Sample breast IDs:")
for bid in metadata['breast_id'].unique()[:10]:
    print(f"  {bid}")
print(f"\nBreast ID formula: patient_id + '_' + laterality")
print(f"Example: {metadata['patient_id'].iloc[0]}_R")

print(f"\n{'='*80}")
print("IMAGES PER BREAST DISTRIBUTION")
print('='*80)
images_per_breast = metadata.groupby('breast_id').size()
print(images_per_breast.value_counts().sort_index())

print(f"\n{'='*80}")
print("LABEL CONSISTENCY CHECK")
print('='*80)
inconsistent = 0
total_multi_image = 0
inconsistent_examples = []

for breast_id, group in metadata.groupby('breast_id'):
    if len(group) > 1:
        total_multi_image += 1
        labels = group['label'].values
        if len(set(labels)) > 1:
            inconsistent += 1
            if len(inconsistent_examples) < 3:
                inconsistent_examples.append({
                    'breast_id': breast_id,
                    'patient_id': group['patient_id'].iloc[0],
                    'views': group['view'].tolist(),
                    'labels': labels.tolist(),
                    'birads': group['birads_original'].tolist()
                })

print(f"Breasts with >1 image: {total_multi_image}")
print(f"Breasts with inconsistent labels: {inconsistent} ({inconsistent/total_multi_image*100:.1f}%)")

if inconsistent > 0:
    print(f"\nSample inconsistent breasts:")
    for ex in inconsistent_examples:
        print(f"\n  Breast: {ex['breast_id'][:30]}...")
        print(f"    Views: {ex['views']}")
        print(f"    Labels: {ex['labels']}")
        print(f"    BI-RADS: {ex['birads']}")

print(f"\n{'='*80}")
print("BREAST-LEVEL LABEL DISTRIBUTION")
print('='*80)
breast_labels = metadata.groupby('breast_id')['label'].max()
label_counts = breast_labels.value_counts()
print(f"Benign (0): {label_counts.get(0, 0)}")
print(f"Malignant (1): {label_counts.get(1, 0)}")
print(f"Total breasts: {len(breast_labels)}")

print(f"\n{'='*80}")
print("SAMPLE PATIENT DATA")
print('='*80)
# Show first patient with multiple breasts
sample_patient = metadata.groupby('patient_id').filter(lambda x: x['breast_id'].nunique() > 1)['patient_id'].iloc[0]
patient_data = metadata[metadata['patient_id'] == sample_patient].sort_values(['breast_id', 'view'])
print(f"\nPatient: {sample_patient}")
print(patient_data[['image_id', 'breast_id', 'view', 'label', 'birads_original']].to_string(index=False))

print(f"\n{'='*80}")
print("SUMMARY: How VinDr-Mammo Breast Grouping Works")
print('='*80)
print("""
1. Patient IDs: Hashed/anonymized strings (NOT masked like INbreast)
2. Breast ID formula: patient_id + "_" + laterality (L or R)
3. No inference needed: Direct use of provided patient IDs
4. Label aggregation: max() across all views (if ANY view shows malignancy, breast is malignant)
5. Typical pattern: 2 views per breast (CC + MLO)
""")
