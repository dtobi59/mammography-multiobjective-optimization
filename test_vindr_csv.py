"""
Test script to verify VinDr CSV parsing works correctly.
"""
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.parsers import VinDrMammoParser

def test_vindr_csv():
    """Test parsing the VinDr CSV file."""

    # Use local CSV file for testing
    csv_path = os.path.join(os.path.dirname(__file__), "vindr_detection_v1_folds.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        return False

    print("=" * 80)
    print("TESTING VINDR CSV PARSING")
    print("=" * 80)
    print(f"CSV file: {csv_path}\n")

    # Check raw CSV structure first
    print("Step 1: Reading raw CSV...")
    df_raw = pd.read_csv(csv_path, nrows=5, low_memory=False)
    print(f"  [OK] Successfully read CSV")
    print(f"  [OK] Total columns: {len(df_raw.columns)}")
    print(f"  [OK] Columns: {', '.join(df_raw.columns[:10])}...")

    # Test parser
    print("\nStep 2: Testing VinDrMammoParser...")
    try:
        parser = VinDrMammoParser(
            metadata_path=csv_path,
            image_dir="images",  # dummy, won't check if exists
            image_id_col=config.VINDR_CONFIG["image_id_col"],
            patient_id_col=config.VINDR_CONFIG["patient_id_col"],
            laterality_col=config.VINDR_CONFIG["laterality_col"],
            view_col=config.VINDR_CONFIG["view_col"],
            birads_col=config.VINDR_CONFIG["birads_col"],
            image_extension=config.VINDR_CONFIG["image_extension"],
        )

        parsed_df = parser.parse()

        print(f"  [OK] Parser successful!")
        print(f"  [OK] Parsed {len(parsed_df)} images")
        print(f"  [OK] Unique patients: {parsed_df['patient_id'].nunique()}")
        print(f"  [OK] Unique breasts: {parsed_df['breast_id'].nunique()}")

        # Check label distribution
        label_counts = parsed_df['label'].value_counts()
        print(f"\n  Label distribution:")
        print(f"    Benign (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(parsed_df)*100:.1f}%)")
        print(f"    Malignant (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(parsed_df)*100:.1f}%)")

        # Show sample rows
        print(f"\n  Sample parsed data:")
        print(parsed_df[['image_id', 'patient_id', 'breast_id', 'view', 'label', 'birads_original']].head(3))

        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n  [ERROR] Parser failed!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("[FAILED] TEST FAILED")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = test_vindr_csv()
    sys.exit(0 if success else 1)
