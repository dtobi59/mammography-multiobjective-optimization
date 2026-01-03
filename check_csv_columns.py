"""
Quick script to check the actual column names in the VinDr CSV file.
"""
import pandas as pd

# Path to the CSV file (update if needed)
csv_path = "/content/drive/MyDrive/kaggle_vindr_data/vindr_detection_v1_folds.csv"

print(f"Reading CSV file: {csv_path}\n")

# Read just the first few rows to see columns
df = pd.read_csv(csv_path, nrows=5, low_memory=False)

print("=" * 80)
print("COLUMN NAMES IN CSV FILE:")
print("=" * 80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "=" * 80)
print("FIRST 5 ROWS:")
print("=" * 80)
print(df.head())

print("\n" + "=" * 80)
print("DATA TYPES:")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("EXPECTED COLUMNS FROM config.py:")
print("=" * 80)
print("  image_id_col: 'image_id'")
print("  patient_id_col: 'study_id'")
print("  laterality_col: 'laterality'")
print("  view_col: 'view_position'")
print("  birads_col: 'breast_birads'")
