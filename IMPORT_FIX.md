# Fixing Import Errors

If you get `ModuleNotFoundError: No module named 'data'` or similar import errors, use one of these solutions:

## Solution 1: Install the Package (Recommended)

Install the project as a package in editable mode:

```bash
cd mammography-multiobjective-optimization
pip install -e .
```

This makes all modules (`data`, `models`, `training`, etc.) importable from anywhere.

## Solution 2: Add Project Root to Python Path

If you don't want to install the package, add the project root to your Python path at the start of your script:

```python
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now imports will work
import config
from data.dataset import BreastCancerDataset
from models.resnet import ResNet50WithPartialFineTuning
```

## Solution 3: Run from Project Root

Always run scripts from the project root directory:

```bash
# Good - run from project root
cd mammography-multiobjective-optimization
python optimization/nsga3_runner.py
python test_correctness.py

# Bad - run from subdirectory
cd mammography-multiobjective-optimization/optimization
python nsga3_runner.py  # This will fail!
```

## Solution 4: For Jupyter Notebooks

If using Jupyter notebooks, add this cell at the beginning:

```python
import sys
import os

# Add parent directory to path if not already there
notebook_dir = os.getcwd()
project_root = os.path.dirname(notebook_dir) if 'notebooks' in notebook_dir else notebook_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

## For Google Colab

The Colab notebook (`colab_tutorial.ipynb`) automatically handles this by:

1. Cloning the repository
2. Changing to the project directory
3. Installing dependencies
4. **Installing the package with `pip install -e .`**

If you're using Colab and still getting import errors:

```python
# Run this cell
!pip install -e .
```

## Verifying the Fix

Test that imports work:

```python
# Test imports
import config
from data.dataset import BreastCancerDataset
from models.resnet import ResNet50WithPartialFineTuning
from training.trainer import Trainer
from optimization.problem import BreastCancerOptimizationProblem

print("✓ All imports successful!")
```

## Why This Happens

Python needs to know where to find your modules. The project has this structure:

```
mammography-multiobjective-optimization/
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── ...
├── models/
│   ├── __init__.py
│   └── ...
└── ...
```

When you do `from data.dataset import ...`, Python looks for a `data` package. It can only find it if:

1. The project root is in Python's search path (`sys.path`)
2. OR the package is installed with `pip install -e .`

## Quick Reference

| Situation | Solution |
|-----------|----------|
| Running locally | `pip install -e .` |
| Running scripts | Run from project root |
| Using Jupyter | Add project root to `sys.path` |
| Using Colab | Already handled by notebook |
| Running tests | `pip install -e .` or run from root |

## Still Having Issues?

1. **Check you're in the project directory:**
   ```bash
   pwd  # Should show path ending in /mammography-multiobjective-optimization
   ls   # Should show config.py, setup.py, requirements.txt
   ```

2. **Check Python can find the modules:**
   ```python
   import sys
   print(sys.path)  # Should include your project directory
   ```

3. **Reinstall the package:**
   ```bash
   pip uninstall mammography-multiobjective-optimization
   pip install -e .
   ```

4. **Check for typos:**
   - Module names are lowercase: `data`, `models`, `training`
   - File names match: `dataset.py`, `resnet.py`, etc.

5. **Check `__init__.py` files exist:**
   ```bash
   ls data/__init__.py
   ls models/__init__.py
   ls training/__init__.py
   ```

---

**Most common fix:** Just run `pip install -e .` from the project root!
