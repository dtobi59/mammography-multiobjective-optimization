#!/usr/bin/env python3
"""
Diagnostic script to categorize failed VinDr-Mammo downloads.

This script tests a sample of failed files to determine the types of errors:
- 403 Forbidden (file doesn't exist on server)
- Network/timeout errors
- Other errors

Usage:
    Run in Google Colab after mounting Drive
"""

import subprocess
import json
from pathlib import Path
from collections import Counter
import time

# Configuration
BASE_DIR = Path('/content/drive/MyDrive/vindr-mammo-stratified')
PROGRESS_FILE = BASE_DIR / 'download_progress.json'
BASE_URL = "https://physionet.org/files/vindr-mammo/1.0.0"

# Credentials (update these)
USERNAME = "t-9dolab@uchicago.edu"
PASSWORD = "1234&abcd@D"

def load_progress():
    """Load progress file."""
    with open(PROGRESS_FILE, 'r') as f:
        return json.load(f)

def test_download(file_path: str, verbose=False):
    """
    Test downloading a single file and return error type.

    Returns:
        str: Error type ('403_forbidden', 'timeout', 'network_error', 'success', 'unknown')
    """
    url = f"{BASE_URL}/{file_path}"
    output_path = BASE_DIR / file_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'aria2c',
        f'--http-user={USERNAME}',
        f'--http-passwd={PASSWORD}',
        '-d', str(output_path.parent),
        '-o', output_path.name,
        '-x', '5',
        '--max-tries=1',
        '--timeout=30',
        '--allow-overwrite=true',
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        output = result.stdout + result.stderr

        if verbose:
            print(f"\nTesting: {file_path}")
            print(f"Output: {output[:500]}")

        # Check for specific errors
        if "status=403" in output or "403 Forbidden" in output:
            return '403_forbidden'
        elif "timed out" in output.lower() or "timeout" in output.lower():
            return 'timeout'
        elif output_path.exists() and output_path.stat().st_size > 0:
            # Download succeeded
            size = output_path.stat().st_size
            if verbose:
                print(f"✅ Success! Size: {size:,} bytes")
            # Clean up test file
            output_path.unlink()
            return 'success'
        elif "network" in output.lower() or "connection" in output.lower():
            return 'network_error'
        else:
            return 'unknown'

    except subprocess.TimeoutExpired:
        return 'timeout'
    except Exception as e:
        if verbose:
            print(f"Exception: {e}")
        return 'unknown'

def main():
    """Run diagnostic on failed files."""
    print("=" * 70)
    print("VinDr-Mammo Failed Downloads Diagnostic")
    print("=" * 70)

    # Load progress
    progress = load_progress()
    failed_files = progress.get('failed_files', [])

    print(f"\n📊 Total failed files: {len(failed_files)}")

    if not failed_files:
        print("✅ No failed files to diagnose!")
        return

    # Test a sample of failed files
    sample_size = min(20, len(failed_files))
    print(f"🔍 Testing sample of {sample_size} files...\n")

    error_counts = Counter()
    sample_indices = list(range(0, len(failed_files), max(1, len(failed_files) // sample_size)))[:sample_size]

    for i, idx in enumerate(sample_indices):
        file_path = failed_files[idx]
        print(f"[{i+1}/{sample_size}] Testing: {file_path[:60]}...")

        error_type = test_download(file_path, verbose=False)
        error_counts[error_type] += 1

        print(f"  Result: {error_type}")

        # Small delay to avoid rate limiting
        if i < sample_size - 1:
            time.sleep(2)

    # Print summary
    print("\n" + "=" * 70)
    print("📊 Diagnostic Summary")
    print("=" * 70)

    total_tested = sum(error_counts.values())

    for error_type, count in error_counts.most_common():
        percentage = (count / total_tested) * 100
        print(f"  {error_type:20s}: {count:3d} ({percentage:5.1f}%)")

    # Extrapolate to full dataset
    print(f"\n📈 Extrapolated to all {len(failed_files)} failed files:")

    for error_type, count in error_counts.items():
        estimated = int((count / total_tested) * len(failed_files))
        print(f"  {error_type:20s}: ~{estimated} files")

    # Recommendations
    print("\n💡 Recommendations:")
    print("=" * 70)

    if error_counts.get('403_forbidden', 0) > total_tested * 0.8:
        print("⚠️  Most files return 403 Forbidden")
        print("   → These files don't exist on PhysioNet's server")
        print("   → Cannot be recovered")
        print("   → Accept current dataset size")
        print(f"   → You have {len(progress.get('downloaded_files', []))} valid files")

    elif error_counts.get('timeout', 0) > total_tested * 0.5:
        print("⚠️  Many timeout errors")
        print("   → PhysioNet server may be slow/overloaded")
        print("   → Try downloading during off-peak hours")
        print("   → Increase timeout values")

    elif error_counts.get('network_error', 0) > total_tested * 0.5:
        print("⚠️  Many network errors")
        print("   → Check internet connection stability")
        print("   → Try downloading from different network")

    elif error_counts.get('success', 0) > 0:
        print("✅ Some files can be recovered!")
        print("   → Retry downloading with longer timeouts")
        print("   → Download during off-peak hours")

    else:
        print("❓ Mixed or unknown errors")
        print("   → Manual investigation needed")
        print("   → Check PhysioNet access permissions")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
