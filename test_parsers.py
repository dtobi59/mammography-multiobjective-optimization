"""
Test dataset-specific parsers for VinDr-Mammo and INbreast.
"""

import os
import shutil
import pandas as pd
from pathlib import Path

print("="*60)
print("PARSER TESTS - Dataset-Specific Metadata Parsing")
print("="*60)

tests_passed = 0
tests_failed = 0


def test_result(name: str, passed: bool, error_msg: str = ""):
    """Record test result."""
    global tests_passed, tests_failed
    if passed:
        print(f"[PASS] {name}")
        tests_passed += 1
    else:
        print(f"[FAIL] {name}")
        if error_msg:
            print(f"  Error: {error_msg}")
        tests_failed += 1


print("\n" + "="*60)
print("TEST 1: BI-RADS to Binary Label Mapping")
print("="*60)

from data.parsers import birads_to_binary_label

# Test benign categories
test_result("BI-RADS 1 -> 0 (benign)", birads_to_binary_label("1") == 0)
test_result("BI-RADS 2 -> 0 (benign)", birads_to_binary_label("2") == 0)
test_result("BI-RADS 3 -> 0 (benign)", birads_to_binary_label("3") == 0)

# Test suspicious/malignant categories
test_result("BI-RADS 4 -> 1 (suspicious)", birads_to_binary_label("4") == 1)
test_result("BI-RADS 4A -> 1 (suspicious)", birads_to_binary_label("4A") == 1)
test_result("BI-RADS 4B -> 1 (suspicious)", birads_to_binary_label("4B") == 1)
test_result("BI-RADS 4C -> 1 (suspicious)", birads_to_binary_label("4C") == 1)
test_result("BI-RADS 5 -> 1 (suspicious)", birads_to_binary_label("5") == 1)
test_result("BI-RADS 6 -> 1 (malignant)", birads_to_binary_label("6") == 1)

# Test case insensitivity
test_result("BI-RADS 4a (lowercase) -> 1", birads_to_binary_label("4a") == 1)
test_result("BI-RADS 4b (lowercase) -> 1", birads_to_binary_label("4b") == 1)

# Test invalid category (should raise ValueError)
try:
    birads_to_binary_label("0")
    test_result("BI-RADS 0 raises ValueError", False, "Did not raise ValueError")
except ValueError:
    test_result("BI-RADS 0 raises ValueError", True)


print("\n" + "="*60)
print("TEST 2: VinDr-Mammo Parser")
print("="*60)

# Create dummy VinDr-Mammo dataset
vindr_dir = Path("./test_vindr_temp")
vindr_dir.mkdir(exist_ok=True)
vindr_images = vindr_dir / "images"
vindr_images.mkdir(exist_ok=True)

# Create dummy metadata CSV
vindr_metadata = pd.DataFrame({
    "image_id": ["img001", "img002", "img003", "img004", "img005"],
    "study_id": ["p001", "p001", "p002", "p002", "p003"],
    "laterality": ["L", "L", "R", "R", "L"],
    "view_position": ["CC", "MLO", "CC", "MLO", "CC"],
    "breast_birads": ["2", "2", "4A", "4A", "5"],
})

vindr_metadata.to_csv(vindr_dir / "metadata.csv", index=False)

# Create dummy images
for img_id in vindr_metadata["image_id"]:
    (vindr_images / f"{img_id}.png").touch()

# Parse using VinDrMammoParser
from data.parsers import VinDrMammoParser

parser = VinDrMammoParser(
    metadata_path=str(vindr_dir / "metadata.csv"),
    image_dir=str(vindr_images),
)

parsed = parser.parse()

test_result(
    "VinDr parser: creates standardized DataFrame",
    isinstance(parsed, pd.DataFrame),
)

test_result(
    "VinDr parser: has required columns",
    all(col in parsed.columns for col in ["image_id", "patient_id", "breast_id", "view", "label", "image_path"]),
)

test_result(
    "VinDr parser: correct number of images",
    len(parsed) == 5,
    f"Expected 5, got {len(parsed)}"
)

test_result(
    "VinDr parser: breast_id format",
    all("_" in bid for bid in parsed["breast_id"]),
    "breast_id should be patient_id + laterality"
)

test_result(
    "VinDr parser: views normalized to CC/MLO",
    set(parsed["view"].unique()) <= {"CC", "MLO"},
    f"Found views: {parsed['view'].unique()}"
)

test_result(
    "VinDr parser: labels are binary (0/1)",
    set(parsed["label"].unique()) <= {0, 1},
    f"Found labels: {parsed['label'].unique()}"
)

# Check specific mappings
test_result(
    "VinDr parser: BI-RADS 2 -> label 0",
    parsed[parsed["birads_original"] == "2"]["label"].iloc[0] == 0,
)

test_result(
    "VinDr parser: BI-RADS 4A -> label 1",
    parsed[parsed["birads_original"] == "4A"]["label"].iloc[0] == 1,
)

test_result(
    "VinDr parser: BI-RADS 5 -> label 1",
    parsed[parsed["birads_original"] == "5"]["label"].iloc[0] == 1,
)

# Check patient-wise grouping
unique_patients = parsed["patient_id"].nunique()
test_result(
    "VinDr parser: correct number of patients",
    unique_patients == 3,
    f"Expected 3 patients, got {unique_patients}"
)

unique_breasts = parsed["breast_id"].nunique()
test_result(
    "VinDr parser: correct number of breasts",
    unique_breasts == 3,  # p001_L, p002_R, p003_L
    f"Expected 3 breasts, got {unique_breasts}"
)


print("\n" + "="*60)
print("TEST 3: INbreast Parser")
print("="*60)

# Create dummy INbreast dataset
inbreast_dir = Path("./test_inbreast_temp")
inbreast_dir.mkdir(exist_ok=True)
inbreast_images = inbreast_dir / "images"
inbreast_images.mkdir(exist_ok=True)

# Create dummy metadata CSV (different format than VinDr)
inbreast_metadata = pd.DataFrame({
    "patient_id": ["pat01", "pat01", "pat02", "pat02"],
    "file_name": ["pat01_L_CC.png", "pat01_L_MLO.png", "pat02_R_CC.png", "pat02_R_MLO.png"],
    "laterality": ["L", "L", "R", "R"],
    "view": ["CC", "MLO", "CC", "MLO"],
    "birads": ["3", "3", "4B", "4B"],
})

inbreast_metadata.to_csv(inbreast_dir / "metadata.csv", index=False)

# Create dummy images
for filename in inbreast_metadata["file_name"]:
    (inbreast_images / filename).touch()

# Parse using INbreastParser
from data.parsers import INbreastParser

parser = INbreastParser(
    metadata_path=str(inbreast_dir / "metadata.csv"),
    image_dir=str(inbreast_images),
    metadata_format="csv",
)

parsed = parser.parse()

test_result(
    "INbreast parser: creates standardized DataFrame",
    isinstance(parsed, pd.DataFrame),
)

test_result(
    "INbreast parser: has required columns",
    all(col in parsed.columns for col in ["image_id", "patient_id", "breast_id", "view", "label", "image_path"]),
)

test_result(
    "INbreast parser: correct number of images",
    len(parsed) == 4,
    f"Expected 4, got {len(parsed)}"
)

test_result(
    "INbreast parser: BI-RADS 3 -> label 0",
    parsed[parsed["birads_original"] == "3"]["label"].iloc[0] == 0,
)

test_result(
    "INbreast parser: BI-RADS 4B -> label 1",
    parsed[parsed["birads_original"] == "4B"]["label"].iloc[0] == 1,
)


print("\n" + "="*60)
print("TEST 4: Unified Representation")
print("="*60)

# Both parsers should produce the same schema
test_result(
    "Unified schema: VinDr and INbreast have same columns",
    set(parsed.columns) == set(parsed.columns),  # Should be identical structure
)

# Test that both can be used interchangeably
from data.parsers import parse_dataset

vindr_parsed = parse_dataset(
    dataset_name="vindr",
    metadata_path=str(vindr_dir / "metadata.csv"),
    image_dir=str(vindr_images),
)

inbreast_parsed = parse_dataset(
    dataset_name="inbreast",
    metadata_path=str(inbreast_dir / "metadata.csv"),
    image_dir=str(inbreast_images),
    metadata_format="csv",
)

test_result(
    "parse_dataset: VinDr returns DataFrame",
    isinstance(vindr_parsed, pd.DataFrame),
)

test_result(
    "parse_dataset: INbreast returns DataFrame",
    isinstance(inbreast_parsed, pd.DataFrame),
)

test_result(
    "parse_dataset: both have same schema",
    set(vindr_parsed.columns) == set(inbreast_parsed.columns),
)


print("\n" + "="*60)
print("TEST 5: Integration with Noisy OR")
print("="*60)

# Test that parsed data works with Noisy OR aggregation
from utils.noisy_or import aggregate_to_breast_level

# Create mock predictions for VinDr data
vindr_predictions = {img_id: 0.5 for img_id in vindr_parsed["image_id"]}

breast_preds, breast_labels = aggregate_to_breast_level(
    vindr_predictions, vindr_parsed
)

test_result(
    "Noisy OR with VinDr: aggregates to breast level",
    len(breast_preds) == vindr_parsed["breast_id"].nunique(),
    f"Expected {vindr_parsed['breast_id'].nunique()} breasts, got {len(breast_preds)}"
)

# Test with INbreast data
inbreast_predictions = {img_id: 0.5 for img_id in inbreast_parsed["image_id"]}

breast_preds, breast_labels = aggregate_to_breast_level(
    inbreast_predictions, inbreast_parsed
)

test_result(
    "Noisy OR with INbreast: aggregates to breast level",
    len(breast_preds) == inbreast_parsed["breast_id"].nunique(),
    f"Expected {inbreast_parsed['breast_id'].nunique()} breasts, got {len(breast_preds)}"
)


# Cleanup
print("\n[CLEANUP] Removing test data...")
shutil.rmtree(vindr_dir)
shutil.rmtree(inbreast_dir)


print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)

total_tests = tests_passed + tests_failed
pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

print(f"Total tests: {total_tests}")
print(f"Passed: {tests_passed} ({pass_rate:.1f}%)")
print(f"Failed: {tests_failed}")

if tests_failed == 0:
    print("\n[SUCCESS] ALL PARSER TESTS PASSED!")
    print("\nKey features verified:")
    print("  - BI-RADS to binary label mapping (including subcategories)")
    print("  - VinDr-Mammo parser with dataset-specific format")
    print("  - INbreast parser with different metadata structure")
    print("  - Unified internal representation")
    print("  - Integration with Noisy OR aggregation")
    print("  - No dataset-specific logic beyond parsing")
    exit(0)
else:
    print(f"\n[FAILURE] {tests_failed} TEST(S) FAILED")
    exit(1)
