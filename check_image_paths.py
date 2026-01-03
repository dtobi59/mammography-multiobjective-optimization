"""Quick script to check image paths are constructed correctly."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.parsers import VinDrMammoParser

csv_path = "vindr_detection_v1_folds.csv"

parser = VinDrMammoParser(
    metadata_path=csv_path,
    image_dir=config.VINDR_CONFIG["image_dir"],
    image_id_col=config.VINDR_CONFIG["image_id_col"],
    patient_id_col=config.VINDR_CONFIG["patient_id_col"],
    laterality_col=config.VINDR_CONFIG["laterality_col"],
    view_col=config.VINDR_CONFIG["view_col"],
    birads_col=config.VINDR_CONFIG["birads_col"],
    image_extension=config.VINDR_CONFIG["image_extension"],
)

df = parser.parse()

print("\nSample image paths:")
print("=" * 80)
for idx, row in df.head(3).iterrows():
    print(f"Patient ID:  {row['patient_id']}")
    print(f"Image ID:    {row['image_id']}")
    print(f"Image Path:  {row['image_path']}")
    print(f"Expected:    {row['patient_id']}/{row['image_id']}.png")
    print("-" * 80)
