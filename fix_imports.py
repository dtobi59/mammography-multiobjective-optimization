"""
Fix imports by adding project root to sys.path.
Import this at the top of every file before any local imports.

Usage:
    import fix_imports  # Must be first import
    import config
    from data.dataset import MammographyDataset
    ...
"""

import sys
import os
from pathlib import Path

# Get the project root directory (parent of this file)
project_root = str(Path(__file__).parent.absolute())

# Add to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also ensure we're in the project directory
os.chdir(project_root)

# Debug info (comment out in production)
# print(f"[fix_imports] Project root: {project_root}")
# print(f"[fix_imports] sys.path[0]: {sys.path[0]}")
