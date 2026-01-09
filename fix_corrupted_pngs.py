"""
Find and fix corrupted PNG files from DICOM conversion.

This script:
1. Scans all PNG files in the images_png directory
2. Attempts to open each one with PIL
3. Identifies corrupted files
4. Reconverts them from the original DICOM
"""

import sys
from pathlib import Path
from PIL import Image
import pydicom
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config


def validate_png(png_path):
    """
    Check if a PNG file is valid and can be opened.

    Args:
        png_path: Path to PNG file

    Returns:
        True if valid, False if corrupted
    """
    try:
        with Image.open(png_path) as img:
            img.load()  # Force load to detect corruption
            return True
    except Exception:
        return False


def convert_dicom_to_png(dicom_path, png_path):
    """
    Convert a single DICOM file to PNG.

    Args:
        dicom_path: Path to DICOM file
        png_path: Path to output PNG file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read DICOM
        dcm = pydicom.dcmread(str(dicom_path))
        img_array = dcm.pixel_array.astype(float)

        # Apply PhotometricInterpretation if needed
        if hasattr(dcm, 'PhotometricInterpretation'):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                img_array = img_array.max() - img_array

        # Normalize to 0-255
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min) * 255.0
        else:
            img_array = np.zeros_like(img_array)

        img_array = img_array.astype(np.uint8)

        # Convert to PIL Image and save
        img = Image.fromarray(img_array, mode='L')
        img.save(str(png_path), "PNG")

        return True
    except Exception as e:
        print(f"  Error converting {dicom_path.name}: {str(e)[:100]}")
        return False


def main():
    """Main function to find and fix corrupted PNGs."""

    # Check if config has INBREAST_PATH
    if not hasattr(config, 'INBREAST_PATH'):
        print("Error: INBREAST_PATH not found in config.py")
        print("Please set up your configuration first.")
        return

    # Set up paths
    inbreast_path = Path(config.INBREAST_PATH)
    dicom_dir = inbreast_path / "AllDICOMs"
    png_dir = inbreast_path / "images_png"

    print("=" * 80)
    print("CORRUPTED PNG FILE DETECTION AND REPAIR")
    print("=" * 80)
    print()
    print(f"DICOM directory: {dicom_dir}")
    print(f"PNG directory: {png_dir}")
    print()

    # Check directories exist
    if not png_dir.exists():
        print(f"Error: PNG directory not found: {png_dir}")
        return

    if not dicom_dir.exists():
        print(f"Error: DICOM directory not found: {dicom_dir}")
        return

    # Find all PNG files
    png_files = list(png_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files to check")
    print()

    # Check each PNG file
    print("Scanning for corrupted files...")
    corrupted_files = []

    for png_path in tqdm(png_files, desc="Validating"):
        if not validate_png(png_path):
            corrupted_files.append(png_path)

    print()
    print("=" * 80)
    print("SCAN RESULTS")
    print("=" * 80)
    print(f"Total PNG files: {len(png_files)}")
    print(f"Valid files: {len(png_files) - len(corrupted_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print()

    if len(corrupted_files) == 0:
        print("[OK] All PNG files are valid!")
        print("=" * 80)
        return

    # Show corrupted files
    print("Corrupted files:")
    for i, png_path in enumerate(corrupted_files[:10], 1):
        print(f"  {i}. {png_path.name}")

    if len(corrupted_files) > 10:
        print(f"  ... and {len(corrupted_files) - 10} more")
    print()

    # Ask user if they want to fix
    print("=" * 80)
    print("REPAIR OPTIONS")
    print("=" * 80)
    print()
    print("Would you like to reconvert these files from DICOM?")
    print("  1. Yes - Reconvert all corrupted files")
    print("  2. No - Just report and exit")
    print()

    choice = input("Enter choice (1 or 2): ").strip()

    if choice != "1":
        print("\nExiting without repairs.")
        print("You can manually delete corrupted files and re-run the conversion cell.")
        return

    # Reconvert corrupted files
    print()
    print("=" * 80)
    print("RECONVERTING CORRUPTED FILES")
    print("=" * 80)
    print()

    success_count = 0
    failed_count = 0

    for png_path in tqdm(corrupted_files, desc="Reconverting"):
        # Find corresponding DICOM file
        dicom_path = dicom_dir / (png_path.stem + ".dcm")

        if not dicom_path.exists():
            # Try uppercase extension
            dicom_path = dicom_dir / (png_path.stem + ".DCM")

        if not dicom_path.exists():
            print(f"\n  [!] DICOM not found for {png_path.name}")
            failed_count += 1
            continue

        # Delete corrupted PNG
        try:
            png_path.unlink()
        except Exception as e:
            print(f"\n  [!] Could not delete {png_path.name}: {e}")
            failed_count += 1
            continue

        # Reconvert
        if convert_dicom_to_png(dicom_path, png_path):
            success_count += 1
        else:
            failed_count += 1

    print()
    print("=" * 80)
    print("REPAIR COMPLETE")
    print("=" * 80)
    print(f"Successfully repaired: {success_count} files")
    print(f"Failed to repair: {failed_count} files")
    print()

    if failed_count > 0:
        print("[!] Some files could not be repaired.")
        print("    These files may have missing or corrupted DICOM sources.")
    else:
        print("[OK] All corrupted files have been repaired!")

    print("=" * 80)


if __name__ == "__main__":
    main()
