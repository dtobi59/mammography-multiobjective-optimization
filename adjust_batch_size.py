"""
Script to adjust batch size in config.py

Use this if you're getting GPU memory errors or want to optimize speed.
"""

import sys

def set_batch_size(new_batch_size: int):
    """
    Update batch size in config.py

    Recommended values:
    - 8: Low memory GPUs (Colab free tier, T4)
    - 16: Standard (current default)
    - 32: High memory GPUs (A100, V100)
    - 64: Very high memory GPUs (multiple GPUs)
    """

    print(f"Setting batch size to {new_batch_size}...")

    # Read config
    with open('config.py', 'r') as f:
        content = f.read()

    # Find and replace batch size
    import re
    pattern = r'BATCH_SIZE = \d+'
    match = re.search(pattern, content)

    if match:
        old_value = match.group()
        new_content = content.replace(old_value, f'BATCH_SIZE = {new_batch_size}')

        # Write back
        with open('config.py', 'w') as f:
            f.write(new_content)

        print(f"✓ Updated: {old_value} → BATCH_SIZE = {new_batch_size}")
        print(f"\nMemory implications:")

        if new_batch_size <= 8:
            print("  ✓ Low memory usage (~2-3 GB)")
            print("  ⚠ Slower training (more batches)")
            print("  ⚠ May have noisier gradients")
        elif new_batch_size <= 16:
            print("  ✓ Balanced memory (~4-6 GB)")
            print("  ✓ Good training stability")
        elif new_batch_size <= 32:
            print("  ✓ Faster training (fewer batches)")
            print("  ⚠ Higher memory (~8-10 GB)")
            print("  ✓ More stable gradients")
        else:
            print("  ⚠ Very high memory (>10 GB)")
            print("  ✓ Fastest training")
            print("  ⚠ May need multiple GPUs")

        return True
    else:
        print("✗ Could not find BATCH_SIZE in config.py")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python adjust_batch_size.py <new_batch_size>")
        print("\nExamples:")
        print("  python adjust_batch_size.py 8   # For low memory")
        print("  python adjust_batch_size.py 16  # Default")
        print("  python adjust_batch_size.py 32  # For high memory GPUs")
        sys.exit(1)

    try:
        new_batch_size = int(sys.argv[1])
        if new_batch_size < 1 or new_batch_size > 128:
            print("Error: Batch size must be between 1 and 128")
            sys.exit(1)

        set_batch_size(new_batch_size)
    except ValueError:
        print("Error: Batch size must be an integer")
        sys.exit(1)
