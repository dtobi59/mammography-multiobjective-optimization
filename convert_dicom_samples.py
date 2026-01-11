"""
Convert the two DICOM sample files to PNG and display them.
"""

import sys
from pathlib import Path
_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    print("Installing pydicom...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom"], check=True)
    import pydicom

def convert_dicom_to_png(dicom_path, png_path):
    """
    Convert a single DICOM file to PNG using the project's pipeline.

    Args:
        dicom_path: Path to DICOM file
        png_path: Path to output PNG file

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Converting: {dicom_path.name}")

        # Read DICOM
        dcm = pydicom.dcmread(str(dicom_path))
        img_array = dcm.pixel_array.astype(float)

        print(f"  Original shape: {img_array.shape}")
        print(f"  Original range: [{img_array.min():.2f}, {img_array.max():.2f}]")

        # Apply PhotometricInterpretation if needed
        if hasattr(dcm, 'PhotometricInterpretation'):
            print(f"  PhotometricInterpretation: {dcm.PhotometricInterpretation}")
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                img_array = img_array.max() - img_array
                print(f"  Applied MONOCHROME1 inversion")

        # Normalize to 0-255
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min) * 255.0
        else:
            img_array = np.zeros_like(img_array)

        img_array = img_array.astype(np.uint8)
        print(f"  Normalized range: [{img_array.min()}, {img_array.max()}]")

        # Convert to PIL Image and save
        img = Image.fromarray(img_array, mode='L')
        img.save(str(png_path), "PNG")

        print(f"  [OK] Saved to: {png_path.name}")
        return True

    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return False


def main():
    """Convert the two DICOM files in the project directory."""

    project_dir = Path(__file__).parent

    # Find DICOM files
    dicom_files = [
        project_dir / "4a06e41e38450e4c6e743b8e8da1ec60.dicom",
        project_dir / "6cab0db67458b7c2bf4afade630adee5.dicom"
    ]

    print("=" * 80)
    print("DICOM to PNG Conversion")
    print("=" * 80)
    print()

    converted_files = []

    for dicom_path in dicom_files:
        if not dicom_path.exists():
            print(f"Warning: {dicom_path.name} not found")
            continue

        # Create PNG filename
        png_path = dicom_path.with_suffix('.png')

        # Convert
        success = convert_dicom_to_png(dicom_path, png_path)

        if success:
            converted_files.append(png_path)

        print()

    print("=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total files converted: {len(converted_files)}")

    for png_file in converted_files:
        print(f"  [OK] {png_file.name}")

    print()
    print("PNG files are ready for viewing!")
    print("=" * 80)

    return converted_files


if __name__ == "__main__":
    main()
