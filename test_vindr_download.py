"""
Test script to download 5 stratified DICOM files from VinDr-Mammo dataset.
Downloads locally to test the stratification and download logic.
"""

import os
import sys
from pathlib import Path
import subprocess
import getpass
import pandas as pd
import numpy as np

_root = str(Path(__file__).parent.absolute())
if _root not in sys.path:
    sys.path.insert(0, _root)


class VinDrMammoTestDownloader:
    """Test downloader for VinDr-Mammo dataset (5 files only)."""

    def __init__(self, output_dir='vindr-mammo-test'):
        """
        Initialize test downloader.

        Args:
            output_dir: Local directory for output
        """
        self.base_dir = Path(output_dir)
        self.base_url = "https://physionet.org/files/vindr-mammo/1.0.0"

        self.username = None
        self.password = None

        # Test parameters
        self.num_files = 5
        self.random_seed = 42

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'images').mkdir(exist_ok=True)
        (self.base_dir / 'metadata').mkdir(exist_ok=True)

        print(f"[OK] Initialized test downloader at {self.base_dir}")

    def setup_credentials(self, username: str = None, password: str = None) -> bool:
        """
        Setup PhysioNet credentials.

        Args:
            username: PhysioNet username
            password: PhysioNet password

        Returns:
            True if credentials are valid
        """
        if not username:
            print("\nPhysioNet Credentials Required")
            print("Get credentials at: https://physionet.org/")
            username = input("Username: ").strip()
            password = getpass.getpass("Password: ")

        self.username = username
        self.password = password

        print("Verifying credentials...")
        return self._test_access()

    def _test_access(self) -> bool:
        """Test PhysioNet access."""
        test_url = f"{self.base_url}/SHA256SUMS.txt"
        test_file = self.base_dir / "test_access.txt"

        cmd = [
            'curl',
            '-u', f'{self.username}:{self.password}',
            '-o', str(test_file),
            '-s', '-S',
            '--max-time', '15',
            '--retry', '2',
            test_url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=20)

            if result.returncode == 0 and test_file.exists() and test_file.stat().st_size > 0:
                test_file.unlink()
                print("[OK] Credentials verified!")
                return True
            else:
                print("[ERROR] Authentication failed!")
                print(f"Error: {result.stderr.decode()}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

    def download_metadata(self) -> bool:
        """Download metadata CSV files."""
        print("\nDownloading Metadata Files")
        print("=" * 70)

        metadata_dir = self.base_dir / 'metadata'
        csv_file = 'breast-level_annotations.csv'

        url = f"{self.base_url}/{csv_file}"
        output_file = metadata_dir / csv_file

        # Check if already exists
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"  [OK] {csv_file} already exists")
            return True

        print(f"  Downloading {csv_file}...", end=" ", flush=True)

        cmd = [
            'curl',
            '-u', f'{self.username}:{self.password}',
            '-o', str(output_file),
            '-s', '-S',
            '--max-time', '60',
            '--retry', '3',
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=90)

            if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"[OK] ({size_mb:.2f} MB)")
                return True
            else:
                print("[ERROR] Failed")
                print(f"Error: {result.stderr.decode()}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

    def perform_stratified_selection(self) -> pd.DataFrame:
        """
        Perform stratified selection of 5 images for testing.

        Returns:
            DataFrame with selected images
        """
        print("\nPerforming Stratified Selection (5 files for testing)")
        print("=" * 70)

        # Load metadata
        csv_file = self.base_dir / 'metadata' / 'breast-level_annotations.csv'

        if not csv_file.exists():
            print("[ERROR] Metadata not found. Run download_metadata() first.")
            return None

        df = pd.read_csv(csv_file)
        print(f"  Total images in dataset: {len(df)}")

        # Extract numeric BI-RADS values
        df['birads_numeric'] = df['breast_birads'].str.extract(r'(\d+)')[0].astype(float)

        # Exclude BI-RADS 3
        df_filtered = df[df['birads_numeric'] != 3].copy()
        print(f"  After excluding BI-RADS 3: {len(df_filtered)} images")

        # Classify as malignant or benign
        df_filtered['label'] = df_filtered['birads_numeric'].apply(
            lambda x: 1 if x in [4, 5, 6] else 0
        )

        malignant_df = df_filtered[df_filtered['label'] == 1]
        benign_df = df_filtered[df_filtered['label'] == 0]

        print(f"\n  Classification:")
        print(f"     Malignant (BI-RADS 4, 5, 6): {len(malignant_df)} images")
        print(f"     Benign (BI-RADS 1, 2): {len(benign_df)} images")

        # For testing: sample 2 malignant, 3 benign (proportional to 250/750)
        num_malignant = 2
        num_benign = 3

        print(f"\n  Sampling {num_malignant} malignant + {num_benign} benign for testing...")

        np.random.seed(self.random_seed)

        malignant_sample = malignant_df.sample(n=min(num_malignant, len(malignant_df)),
                                                random_state=self.random_seed)
        benign_sample = benign_df.sample(n=min(num_benign, len(benign_df)),
                                          random_state=self.random_seed)

        # Combine samples
        selected_df = pd.concat([malignant_sample, benign_sample], ignore_index=True)

        print(f"\n  [OK] Selection Complete:")
        print(f"     Malignant images: {len(malignant_sample)}")
        print(f"     Benign images: {len(benign_sample)}")
        print(f"     Total images: {len(selected_df)}")

        # Show selected files
        print(f"\n  Selected files:")
        for idx, row in selected_df.iterrows():
            birads = row['breast_birads']
            label_str = "malignant" if row['label'] == 1 else "benign"
            print(f"     {row['study_id']}/{row['image_id']}.dicom (BI-RADS {birads}, {label_str})")

        # Save selection
        selection_file = self.base_dir / 'metadata' / 'selected_files.csv'
        selected_df.to_csv(selection_file, index=False)
        print(f"\n  [OK] Selection saved to: {selection_file}")

        return selected_df

    def download_selected_files(self, selected_df: pd.DataFrame) -> bool:
        """
        Download selected files.

        Args:
            selected_df: DataFrame with selected images

        Returns:
            True if successful
        """
        print("\nDownloading Selected Files")
        print("=" * 70)

        success_count = 0
        fail_count = 0

        # Download each file
        for idx, row in selected_df.iterrows():
            study_id = row['study_id']
            image_id = row['image_id']
            birads = row.get('breast_birads', 'Unknown')
            label = "malignant" if row.get('label', -1) == 1 else "benign"

            image_path = f"images/{study_id}/{image_id}.dicom"

            url = f"{self.base_url}/{image_path}"
            output_file = self.base_dir / image_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{idx + 1}/{len(selected_df)}] Downloading {image_id}.dicom (BI-RADS {birads}, {label})...")

            # Download with retry
            if self._download_file(url, output_file):
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"       [OK] {size_mb:.2f} MB")
                success_count += 1
            else:
                print(f"       [ERROR] Failed")
                fail_count += 1

        print(f"\n{'=' * 70}")
        print(f"[OK] Download Complete!")
        print(f"   Downloaded: {success_count}")
        print(f"   Failed: {fail_count}")
        print(f"{'=' * 70}\n")

        return fail_count == 0

    def _download_file(self, url: str, output_file: Path, max_retries: int = 3) -> bool:
        """Download single file with retry."""
        for attempt in range(max_retries):
            cmd = [
                'curl',
                '-u', f'{self.username}:{self.password}',
                '-o', str(output_file),
                '-L', '-s', '-S',
                '--max-time', '60',
                '--retry', '2',
                url
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=90)

                if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
                    return True

                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

            except Exception:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

        return False


def main():
    """Run test download."""
    import argparse

    parser = argparse.ArgumentParser(description='Download 5 stratified VinDr-Mammo files for testing')
    parser.add_argument('--username', help='PhysioNet username (or set PHYSIONET_USERNAME env var)')
    parser.add_argument('--password', help='PhysioNet password (or set PHYSIONET_PASSWORD env var)')
    args = parser.parse_args()

    # Get credentials from args or environment
    username = args.username or os.environ.get('PHYSIONET_USERNAME')
    password = args.password or os.environ.get('PHYSIONET_PASSWORD')

    print("=" * 70)
    print("VinDr-Mammo Test Downloader")
    print("Downloads 5 stratified files for testing")
    print("=" * 70)

    # Initialize downloader
    downloader = VinDrMammoTestDownloader('vindr-mammo-test')

    # Setup credentials
    if not downloader.setup_credentials(username, password):
        print("\n[ERROR] Failed to verify credentials")
        return

    # Download metadata
    if not downloader.download_metadata():
        print("\n[ERROR] Failed to download metadata")
        return

    # Perform stratified selection
    selected_df = downloader.perform_stratified_selection()
    if selected_df is None:
        print("\n[ERROR] Failed to perform selection")
        return

    # Download selected files
    success = downloader.download_selected_files(selected_df)

    if success:
        print("\n[OK] Test download completed successfully!")
        print(f"\nFiles saved to: {downloader.base_dir / 'images'}")
    else:
        print("\n[WARNING] Some downloads failed")


if __name__ == "__main__":
    main()
