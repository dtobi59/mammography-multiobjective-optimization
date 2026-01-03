"""
Dataset Visualization Script

Visualizes VinDr-Mammo and INbreast datasets before training.
Shows:
- Dataset statistics
- Label distributions
- BI-RADS distributions
- View distributions
- Sample images with labels
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.parsers import VinDrMammoParser, INbreastParser

# Set style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)


def load_datasets():
    """Load both VinDr and INbreast datasets."""

    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    # Load VinDr-Mammo
    print("\n1. Loading VinDr-Mammo dataset...")
    vindr_csv = os.path.join(os.path.dirname(__file__), "vindr_detection_v1_folds.csv")

    if os.path.exists(vindr_csv):
        vindr_parser = VinDrMammoParser(
            metadata_path=vindr_csv,
            image_dir="images",
            image_id_col=config.VINDR_CONFIG["image_id_col"],
            patient_id_col=config.VINDR_CONFIG["patient_id_col"],
            laterality_col=config.VINDR_CONFIG["laterality_col"],
            view_col=config.VINDR_CONFIG["view_col"],
            birads_col=config.VINDR_CONFIG["birads_col"],
            image_extension=config.VINDR_CONFIG["image_extension"],
        )
        vindr_df = vindr_parser.parse()
        print(f"   [OK] Loaded {len(vindr_df)} images")
    else:
        print(f"   [WARNING] VinDr CSV not found at {vindr_csv}")
        vindr_df = None

    # Load INbreast
    print("\n2. Loading INbreast dataset...")
    inbreast_csv = os.path.join(os.path.dirname(__file__), "INbreast.csv")

    if os.path.exists(inbreast_csv):
        inbreast_parser = INbreastParser(
            metadata_path=inbreast_csv,
            image_dir="images",
            metadata_format=config.INBREAST_CONFIG["metadata_format"],
            patient_id_col=config.INBREAST_CONFIG["patient_id_col"],
            laterality_col=config.INBREAST_CONFIG["laterality_col"],
            view_col=config.INBREAST_CONFIG["view_col"],
            birads_col=config.INBREAST_CONFIG["birads_col"],
            filename_col=config.INBREAST_CONFIG["filename_col"],
        )
        inbreast_df = inbreast_parser.parse()
        print(f"   [OK] Loaded {len(inbreast_df)} images")
    else:
        print(f"   [WARNING] INbreast CSV not found at {inbreast_csv}")
        inbreast_df = None

    print("\n" + "=" * 80)

    return vindr_df, inbreast_df


def plot_dataset_overview(vindr_df, inbreast_df):
    """Plot overview statistics for both datasets."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')

    datasets = [
        ('VinDr-Mammo (Source)', vindr_df, '#2ecc71'),
        ('INbreast (Target)', inbreast_df, '#e74c3c')
    ]

    for idx, (name, df, color) in enumerate(datasets):
        if df is None:
            continue

        row = idx

        # 1. Label Distribution
        ax = axes[row, 0]
        label_counts = df['label'].value_counts().sort_index()
        bars = ax.bar(['Benign (0)', 'Malignant (1)'],
                      [label_counts.get(0, 0), label_counts.get(1, 0)],
                      color=[color, '#95a5a6'], alpha=0.7, edgecolor='black')
        ax.set_title(f'{name}\nLabel Distribution', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

        # Add percentage labels
        total = len(df)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height}\n({height/total*100:.1f}%)',
                   ha='center', va='bottom')

        # 2. BI-RADS Distribution
        ax = axes[row, 1]
        birads_counts = df['birads_original'].value_counts().sort_index()
        birads_labels = [str(b).replace('BI-RADS ', '') for b in birads_counts.index]
        bars = ax.bar(range(len(birads_counts)), birads_counts.values,
                      color=color, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(birads_counts)))
        ax.set_xticklabels(birads_labels, rotation=45, ha='right')
        ax.set_title(f'{name}\nBI-RADS Distribution', fontweight='bold')
        ax.set_xlabel('BI-RADS Category')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

        # Color code by benign/malignant
        for i, (birads, count) in enumerate(birads_counts.items()):
            label = df[df['birads_original'] == birads]['label'].iloc[0]
            bars[i].set_color('#2ecc71' if label == 0 else '#e74c3c')

        # 3. View Distribution
        ax = axes[row, 2]
        view_counts = df['view'].value_counts()
        ax.pie(view_counts.values, labels=view_counts.index, autopct='%1.1f%%',
               colors=[color, '#95a5a6'], startangle=90)
        ax.set_title(f'{name}\nView Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight')
    print("\n[SAVED] dataset_overview.png")
    plt.show()


def plot_detailed_statistics(vindr_df, inbreast_df):
    """Plot detailed statistics."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Dataset Statistics', fontsize=16, fontweight='bold')

    # 1. Images per Patient (VinDr only - INbreast has few patients)
    if vindr_df is not None:
        ax = axes[0, 0]
        images_per_patient = vindr_df.groupby('patient_id').size()
        ax.hist(images_per_patient, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_title('VinDr-Mammo: Images per Patient', fontweight='bold')
        ax.set_xlabel('Number of Images')
        ax.set_ylabel('Number of Patients')
        ax.axvline(images_per_patient.mean(), color='red', linestyle='--',
                   label=f'Mean: {images_per_patient.mean():.1f}')
        ax.legend()
        ax.grid(alpha=0.3)

    # 2. Label Distribution by View (VinDr)
    if vindr_df is not None:
        ax = axes[0, 1]
        view_label_counts = vindr_df.groupby(['view', 'label']).size().unstack(fill_value=0)
        view_label_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'],
                               alpha=0.7, edgecolor='black')
        ax.set_title('VinDr-Mammo: Label Distribution by View', fontweight='bold')
        ax.set_xlabel('View')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(['Benign', 'Malignant'])
        ax.grid(axis='y', alpha=0.3)

    # 3. Label Distribution by View (INbreast)
    if inbreast_df is not None:
        ax = axes[1, 0]
        view_label_counts = inbreast_df.groupby(['view', 'label']).size().unstack(fill_value=0)
        view_label_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'],
                               alpha=0.7, edgecolor='black')
        ax.set_title('INbreast: Label Distribution by View', fontweight='bold')
        ax.set_xlabel('View')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(['Benign', 'Malignant'])
        ax.grid(axis='y', alpha=0.3)

    # 4. Comparison Table
    ax = axes[1, 1]
    ax.axis('off')

    # Create comparison table
    comparison_data = []

    if vindr_df is not None:
        comparison_data.append([
            'VinDr-Mammo',
            len(vindr_df),
            vindr_df['patient_id'].nunique(),
            vindr_df['breast_id'].nunique(),
            f"{(vindr_df['label'] == 0).sum()} ({(vindr_df['label'] == 0).sum()/len(vindr_df)*100:.1f}%)",
            f"{(vindr_df['label'] == 1).sum()} ({(vindr_df['label'] == 1).sum()/len(vindr_df)*100:.1f}%)",
        ])

    if inbreast_df is not None:
        comparison_data.append([
            'INbreast',
            len(inbreast_df),
            inbreast_df['patient_id'].nunique(),
            inbreast_df['breast_id'].nunique(),
            f"{(inbreast_df['label'] == 0).sum()} ({(inbreast_df['label'] == 0).sum()/len(inbreast_df)*100:.1f}%)",
            f"{(inbreast_df['label'] == 1).sum()} ({(inbreast_df['label'] == 1).sum()/len(inbreast_df)*100:.1f}%)",
        ])

    table = ax.table(cellText=comparison_data,
                     colLabels=['Dataset', 'Images', 'Patients', 'Breasts', 'Benign', 'Malignant'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.3, 1, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Dataset Comparison', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('detailed_statistics.png', dpi=300, bbox_inches='tight')
    print("[SAVED] detailed_statistics.png")
    plt.show()


def print_summary_statistics(vindr_df, inbreast_df):
    """Print detailed summary statistics."""

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if vindr_df is not None:
        print("\n### VinDr-Mammo Dataset ###")
        print(f"Total images:      {len(vindr_df):,}")
        print(f"Unique patients:   {vindr_df['patient_id'].nunique():,}")
        print(f"Unique breasts:    {vindr_df['breast_id'].nunique():,}")
        print(f"\nLabel Distribution:")
        print(f"  Benign (0):      {(vindr_df['label'] == 0).sum():,} ({(vindr_df['label'] == 0).sum()/len(vindr_df)*100:.1f}%)")
        print(f"  Malignant (1):   {(vindr_df['label'] == 1).sum():,} ({(vindr_df['label'] == 1).sum()/len(vindr_df)*100:.1f}%)")
        print(f"\nView Distribution:")
        for view, count in vindr_df['view'].value_counts().items():
            print(f"  {view:3s}:             {count:,} ({count/len(vindr_df)*100:.1f}%)")
        print(f"\nBI-RADS Categories: {vindr_df['birads_original'].nunique()}")

    if inbreast_df is not None:
        print("\n### INbreast Dataset ###")
        print(f"Total images:      {len(inbreast_df):,}")
        print(f"Unique patients:   {inbreast_df['patient_id'].nunique():,}")
        print(f"Unique breasts:    {inbreast_df['breast_id'].nunique():,}")
        print(f"\nLabel Distribution:")
        print(f"  Benign (0):      {(inbreast_df['label'] == 0).sum():,} ({(inbreast_df['label'] == 0).sum()/len(inbreast_df)*100:.1f}%)")
        print(f"  Malignant (1):   {(inbreast_df['label'] == 1).sum():,} ({(inbreast_df['label'] == 1).sum()/len(inbreast_df)*100:.1f}%)")
        print(f"\nView Distribution:")
        for view, count in inbreast_df['view'].value_counts().items():
            print(f"  {view:3s}:             {count:,} ({count/len(inbreast_df)*100:.1f}%)")
        print(f"\nBI-RADS Categories: {inbreast_df['birads_original'].nunique()}")
        print(f"BI-RADS breakdown:")
        for birads in sorted(inbreast_df['birads_original'].unique()):
            count = (inbreast_df['birads_original'] == birads).sum()
            label = inbreast_df[inbreast_df['birads_original'] == birads]['label'].iloc[0]
            print(f"  {birads}: {count} images (Label {label})")

    print("\n" + "=" * 80)


def main():
    """Main visualization function."""

    print("\n" + "=" * 80)
    print("DATASET VISUALIZATION")
    print("=" * 80)
    print("\nThis script visualizes both VinDr-Mammo and INbreast datasets.")
    print("It will generate:")
    print("  1. Dataset overview plots")
    print("  2. Detailed statistics plots")
    print("  3. Summary statistics (printed)")
    print("\n" + "=" * 80)

    # Load datasets
    vindr_df, inbreast_df = load_datasets()

    if vindr_df is None and inbreast_df is None:
        print("\n[ERROR] No datasets found! Please ensure CSV files are in the project directory.")
        return

    # Print summary statistics
    print_summary_statistics(vindr_df, inbreast_df)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_dataset_overview(vindr_df, inbreast_df)
    plot_detailed_statistics(vindr_df, inbreast_df)

    print("\n" + "=" * 80)
    print("[SUCCESS] Visualization complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - dataset_overview.png")
    print("  - detailed_statistics.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
