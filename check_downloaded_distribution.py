#!/usr/bin/env python3
"""
Check the breast-level distribution of downloaded VinDr-Mammo files.

This script verifies:
1. How many malignant vs benign files were downloaded
2. If the 25% malignant / 75% benign target is maintained
3. Patient-level and breast-level distributions
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

# Paths (update these for your environment)
BASE_DIR = Path('/content/drive/MyDrive/vindr-mammo-stratified')
PROGRESS_FILE = BASE_DIR / 'download_progress.json'
SELECTION_FILE = BASE_DIR / 'metadata' / 'selected_files.csv'
METADATA_FILE = BASE_DIR / 'metadata' / 'breast-level_annotations.csv'

def analyze_downloaded_distribution():
    """Analyze the distribution of downloaded files."""

    print("=" * 80)
    print("📊 DOWNLOADED FILES DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Load progress
    if not PROGRESS_FILE.exists():
        print("❌ Progress file not found")
        return

    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)

    downloaded_files = progress.get('downloaded_files', [])

    if not downloaded_files:
        print("❌ No downloaded files found")
        return

    print(f"\n✅ Total downloaded files: {len(downloaded_files)}")

    # Load selection to get labels
    if not SELECTION_FILE.exists():
        print("❌ Selection file not found")
        return

    selection_df = pd.read_csv(SELECTION_FILE)

    # Create mapping of image_path to label
    selection_df['image_path'] = selection_df.apply(
        lambda row: f"images/{row['study_id']}/{row['image_id']}.dicom",
        axis=1
    )

    # Filter to only downloaded files
    downloaded_df = selection_df[selection_df['image_path'].isin(downloaded_files)].copy()

    print(f"✅ Matched {len(downloaded_df)} files with metadata")

    if len(downloaded_df) == 0:
        print("❌ Could not match downloaded files with selection metadata")
        return

    # Calculate label distribution
    print("\n" + "=" * 80)
    print("🎯 LABEL DISTRIBUTION (Downloaded Files)")
    print("=" * 80)

    label_counts = downloaded_df['label'].value_counts()
    total = len(downloaded_df)

    malignant = label_counts.get(1, 0)
    benign = label_counts.get(0, 0)

    malignant_pct = (malignant / total * 100) if total > 0 else 0
    benign_pct = (benign / total * 100) if total > 0 else 0

    print(f"\n  Malignant (label=1): {malignant:4d} ({malignant_pct:5.1f}%)")
    print(f"  Benign (label=0):    {benign:4d} ({benign_pct:5.1f}%)")
    print(f"  Total:               {total:4d}")

    print(f"\n  Target distribution: 25% malignant / 75% benign")
    print(f"  Actual distribution: {malignant_pct:.1f}% malignant / {benign_pct:.1f}% benign")

    # Check if close to target
    malignant_diff = abs(malignant_pct - 25.0)
    benign_diff = abs(benign_pct - 75.0)

    if malignant_diff < 5 and benign_diff < 5:
        print(f"\n  ✅ GOOD: Within 5% of target distribution")
    elif malignant_diff < 10 and benign_diff < 10:
        print(f"\n  ⚠️  ACCEPTABLE: Within 10% of target distribution")
    else:
        print(f"\n  ❌ WARNING: More than 10% deviation from target")

    # Patient-level analysis
    print("\n" + "=" * 80)
    print("👥 PATIENT-LEVEL ANALYSIS")
    print("=" * 80)

    unique_patients = downloaded_df['study_id'].nunique()
    malignant_patients = downloaded_df[downloaded_df['label'] == 1]['study_id'].nunique()
    benign_patients = downloaded_df[downloaded_df['label'] == 0]['study_id'].nunique()

    print(f"\n  Total unique patients: {unique_patients}")
    print(f"  Malignant patients:    {malignant_patients}")
    print(f"  Benign patients:       {benign_patients}")

    # Images per patient
    images_per_patient = downloaded_df.groupby('study_id').size()
    print(f"\n  Images per patient:")
    print(f"    Mean:   {images_per_patient.mean():.1f}")
    print(f"    Median: {images_per_patient.median():.1f}")
    print(f"    Min:    {images_per_patient.min()}")
    print(f"    Max:    {images_per_patient.max()}")

    # BI-RADS distribution
    print("\n" + "=" * 80)
    print("🏥 BI-RADS DISTRIBUTION")
    print("=" * 80)

    birads_counts = downloaded_df['breast_birads'].value_counts().sort_index()
    print("\n  BI-RADS Category | Count | Percentage")
    print("  " + "-" * 40)

    for birads, count in birads_counts.items():
        pct = (count / total * 100)
        label = "Malignant" if birads in ['BI-RADS 4', 'BI-RADS 5', 'BI-RADS 6'] else "Benign"
        print(f"  {birads:15s} | {count:5d} | {pct:5.1f}% ({label})")

    # Breast laterality distribution
    print("\n" + "=" * 80)
    print("🔄 LATERALITY DISTRIBUTION")
    print("=" * 80)

    if 'laterality' in downloaded_df.columns:
        laterality_counts = downloaded_df['laterality'].value_counts()
        print("\n  Side | Count | Percentage")
        print("  " + "-" * 30)

        for side, count in laterality_counts.items():
            pct = (count / total * 100)
            print(f"  {side:4s} | {count:5d} | {pct:5.1f}%")

    # View position distribution
    print("\n" + "=" * 80)
    print("📸 VIEW POSITION DISTRIBUTION")
    print("=" * 80)

    if 'view_position' in downloaded_df.columns:
        view_counts = downloaded_df['view_position'].value_counts()
        print("\n  View | Count | Percentage")
        print("  " + "-" * 30)

        for view, count in view_counts.items():
            pct = (count / total * 100)
            print(f"  {view:4s} | {count:5d} | {pct:5.1f}%")

    # Comparison with original selection
    print("\n" + "=" * 80)
    print("📊 COMPARISON: Original Selection vs Downloaded")
    print("=" * 80)

    orig_total = len(selection_df)
    orig_malignant = (selection_df['label'] == 1).sum()
    orig_benign = (selection_df['label'] == 0).sum()

    print(f"\n  Original Selection:")
    print(f"    Total:     {orig_total:4d}")
    print(f"    Malignant: {orig_malignant:4d} ({orig_malignant/orig_total*100:5.1f}%)")
    print(f"    Benign:    {orig_benign:4d} ({orig_benign/orig_total*100:5.1f}%)")

    print(f"\n  Downloaded:")
    print(f"    Total:     {total:4d} ({total/orig_total*100:5.1f}% of selection)")
    print(f"    Malignant: {malignant:4d} ({malignant_pct:5.1f}%)")
    print(f"    Benign:    {benign:4d} ({benign_pct:5.1f}%)")

    # Final verdict
    print("\n" + "=" * 80)
    print("✅ FINAL VERDICT")
    print("=" * 80)

    if malignant_diff < 5 and benign_diff < 5 and total >= 200:
        print("\n  ✅ EXCELLENT: Good representation with valid stratification")
        print(f"     - {total} files is sufficient for experimentation")
        print(f"     - Stratification is well-maintained")
        print(f"     - Proceed with model training")
    elif malignant_diff < 10 and benign_diff < 10 and total >= 100:
        print("\n  ✅ GOOD: Acceptable representation")
        print(f"     - {total} files is usable for development/testing")
        print(f"     - Stratification is reasonably maintained")
        print(f"     - May need more data for final model")
    else:
        print("\n  ⚠️  WARNING: Limited representation")
        print(f"     - {total} files may be too small")
        print(f"     - Stratification may be skewed")
        print(f"     - Consider requesting more data or alternative datasets")

    print("\n" + "=" * 80)

    return downloaded_df

if __name__ == "__main__":
    analyze_downloaded_distribution()
