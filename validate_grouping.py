"""
Validate patient grouping strategy for INbreast dataset.
Compare different grouping approaches to find the most accurate one.
"""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd
import numpy as np

def analyze_grouping_strategies():
    """Compare different patient grouping strategies."""

    # Load raw CSV
    csv_path = Path(__file__).parent / "INbreast.csv"
    # Try comma first (updated format), fallback to semicolon
    try:
        df = pd.read_csv(csv_path, sep=',')
    except:
        df = pd.read_csv(csv_path, sep=';')

    print("=" * 80)
    print("PATIENT GROUPING VALIDATION")
    print("=" * 80)
    print(f"\nTotal images: {len(df)}")
    print(f"\nAvailable columns: {list(df.columns)}")

    # Strategy 1: Current approach (file number gaps)
    print("\n" + "=" * 80)
    print("STRATEGY 1: File Number Proximity (gap > 500)")
    print("=" * 80)

    file_numbers = df['File Name'].astype(int)
    sorted_indices = file_numbers.argsort()
    sorted_files = file_numbers.iloc[sorted_indices].values

    patient_groups_v1 = []
    current_patient = 0
    prev_file = sorted_files[0]

    for file_num in sorted_files:
        if abs(file_num - prev_file) > 500:
            current_patient += 1
        patient_groups_v1.append(current_patient)
        prev_file = file_num

    # Reorder
    patient_groups_v1_ordered = [None] * len(df)
    for i, orig_idx in enumerate(sorted_indices):
        patient_groups_v1_ordered[orig_idx] = patient_groups_v1[i]

    df['patient_v1'] = patient_groups_v1_ordered
    df['breast_v1'] = df['patient_v1'].astype(str) + "_" + df['Laterality'].astype(str)

    print(f"Patients identified: {df['patient_v1'].nunique()}")
    print(f"Breasts identified: {df['breast_v1'].nunique()}")
    print(f"Images per breast (avg): {len(df) / df['breast_v1'].nunique():.2f}")

    # Check distribution
    images_per_breast_v1 = df.groupby('breast_v1').size()
    print(f"\nImages per breast distribution:")
    print(images_per_breast_v1.value_counts().sort_index())

    # Strategy 2: Acquisition date grouping
    print("\n" + "=" * 80)
    print("STRATEGY 2: Acquisition Date Grouping")
    print("=" * 80)

    # Group by acquisition date
    date_groups = df.groupby('Acquisition date').groups
    print(f"Unique acquisition dates: {len(date_groups)}")

    df['patient_v2'] = df.groupby('Acquisition date').ngroup()
    df['breast_v2'] = df['patient_v2'].astype(str) + "_" + df['Laterality'].astype(str)

    print(f"Patients identified: {df['patient_v2'].nunique()}")
    print(f"Breasts identified: {df['breast_v2'].nunique()}")
    print(f"Images per breast (avg): {len(df) / df['breast_v2'].nunique():.2f}")

    # Check distribution
    images_per_breast_v2 = df.groupby('breast_v2').size()
    print(f"\nImages per breast distribution:")
    print(images_per_breast_v2.value_counts().sort_index())

    # Strategy 3: Acquisition date + small file gaps
    print("\n" + "=" * 80)
    print("STRATEGY 3: Acquisition Date + File Number Proximity (gap > 100)")
    print("=" * 80)

    # Group by date first, then by file gaps within each date
    df_sorted = df.sort_values(['Acquisition date', 'File Name'])

    patient_groups_v3 = []
    current_patient = 0
    prev_date = None
    prev_file = None

    for _, row in df_sorted.iterrows():
        curr_date = row['Acquisition date']
        curr_file = int(row['File Name'])

        # New patient if date changes OR large file gap
        if prev_date is None:
            pass  # First row
        elif curr_date != prev_date:
            current_patient += 1
        elif abs(curr_file - prev_file) > 100:
            current_patient += 1

        patient_groups_v3.append(current_patient)
        prev_date = curr_date
        prev_file = curr_file

    df_sorted['patient_v3'] = patient_groups_v3
    df = df.merge(df_sorted[['File Name', 'patient_v3']], on='File Name', how='left')
    df['breast_v3'] = df['patient_v3'].astype(str) + "_" + df['Laterality'].astype(str)

    print(f"Patients identified: {df['patient_v3'].nunique()}")
    print(f"Breasts identified: {df['breast_v3'].nunique()}")
    print(f"Images per breast (avg): {len(df) / df['breast_v3'].nunique():.2f}")

    # Check distribution
    images_per_breast_v3 = df.groupby('breast_v3').size()
    print(f"\nImages per breast distribution:")
    print(images_per_breast_v3.value_counts().sort_index())

    # Analyze label consistency within breasts
    print("\n" + "=" * 80)
    print("LABEL CONSISTENCY ANALYSIS")
    print("=" * 80)

    from data.parsers import birads_to_binary_label
    df['label'] = df['Bi-Rads'].apply(lambda x: birads_to_binary_label(str(x)))

    for strategy, breast_col in [('Strategy 1 (gap>500)', 'breast_v1'),
                                   ('Strategy 2 (date)', 'breast_v2'),
                                   ('Strategy 3 (date+gap>100)', 'breast_v3')]:
        print(f"\n{strategy}:")

        # Check how many breasts have inconsistent labels across views
        inconsistent = 0
        total_multi_image = 0

        for breast_id, group in df.groupby(breast_col):
            if len(group) > 1:
                total_multi_image += 1
                labels = group['label'].values
                if len(set(labels)) > 1:
                    inconsistent += 1

        print(f"  Breasts with >1 image: {total_multi_image}")
        print(f"  Breasts with inconsistent labels: {inconsistent} ({inconsistent/total_multi_image*100:.1f}%)")

        # Check breast-level label distribution
        breast_labels = df.groupby(breast_col)['label'].max()
        print(f"  Breast-level labels: {breast_labels.value_counts().to_dict()}")

    # Sample analysis
    print("\n" + "=" * 80)
    print("SAMPLE PATIENT GROUPS (First 5 patients, Strategy 1)")
    print("=" * 80)

    for patient_id in sorted(df['patient_v1'].unique())[:5]:
        patient_data = df[df['patient_v1'] == patient_id].sort_values('File Name')
        print(f"\nPatient {patient_id}:")
        print(patient_data[['File Name', 'Laterality', 'View', 'Acquisition date', 'Bi-Rads', 'label']])

    return df


if __name__ == "__main__":
    df = analyze_grouping_strategies()
