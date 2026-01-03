# Data Directory - Explained

## Question: "I cannot find the data dir on GitHub"

**Answer:** You CAN find it! But it contains **Python code**, not actual datasets.

## What's on GitHub

### ✅ Python Code Modules (All Present)

```
mammography-multiobjective-optimization/
├── data/                    ← YES, this IS on GitHub!
│   ├── __init__.py         ← Python module initialization
│   ├── augmentation.py     ← Image augmentation code
│   ├── dataset.py          ← Dataset loading code
│   └── parsers.py          ← VinDr/INbreast parsers
├── models/                  ← YES, on GitHub
│   ├── __init__.py
│   └── resnet.py           ← ResNet-50 model code
├── training/                ← YES, on GitHub
│   ├── __init__.py
│   ├── trainer.py          ← Training loop code
│   ├── metrics.py          ← Metrics computation code
│   └── robustness.py       ← Robustness evaluation code
├── optimization/            ← YES, on GitHub
│   ├── __init__.py
│   ├── problem.py          ← NSGA-III problem definition
│   ├── nsga3_runner.py     ← Optimization runner
│   └── analyze_pareto.py   ← Results analysis
├── evaluation/              ← YES, on GitHub
│   ├── __init__.py
│   ├── evaluate_source.py  ← Source evaluation code
│   └── evaluate_target.py  ← Target evaluation code
└── utils/                   ← YES, on GitHub
    ├── __init__.py
    ├── seed.py             ← Random seed utilities
    └── noisy_or.py         ← Noisy OR aggregation
```

**View on GitHub:**
https://github.com/dtobi59/mammography-multiobjective-optimization

Click on the `data/` folder and you'll see:
- `__init__.py`
- `augmentation.py`
- `dataset.py`
- `parsers.py`

## What's NOT on GitHub (Intentional)

### ❌ Actual Dataset Image Files

These are **intentionally excluded** from GitHub:

```
❌ vindr_mammo/          (~160 GB - too large for GitHub)
   ├── images/
   │   ├── 00001.png
   │   ├── 00002.png
   │   └── ...
   └── metadata.csv

❌ inbreast/             (~5 GB - too large for GitHub)
   ├── images/
   │   ├── INB_001.png
   │   ├── INB_002.png
   │   └── ...
   └── metadata.csv

❌ demo_data/            (Created locally/in Colab)
   ├── vindr/
   └── inbreast/
```

### Why They're Not on GitHub

1. **Size limits:** GitHub has file size restrictions
2. **Copyright:** Datasets require credentialed access
3. **Best practice:** Data should be downloaded separately
4. **`.gitignore`:** Prevents accidental upload of large files

From `.gitignore`:
```gitignore
# Data directories (don't commit large datasets)
data/
datasets/
vindr_mammo/
inbreast/
*/images/
*/metadata.csv
```

## How to Get the Data

### Option 1: Colab Demo Data (Easiest - No Download)

The Colab notebook **automatically creates synthetic demo data**:

```python
# Runs automatically in Cell 5 of colab_tutorial.ipynb
os.makedirs("demo_data/vindr/images", exist_ok=True)
os.makedirs("demo_data/inbreast/images", exist_ok=True)

# Creates 20 synthetic VinDr images
# Creates 12 synthetic INbreast images
# Perfect for testing the pipeline!
```

**Result:**
```
demo_data/
├── vindr/
│   ├── images/
│   │   ├── P001_L_CC.png
│   │   ├── P001_L_MLO.png
│   │   └── ... (20 total)
│   └── metadata.csv
└── inbreast/
    ├── images/
    │   ├── INbreast_001_L_CC.png
    │   └── ... (12 total)
    └── metadata.csv
```

### Option 2: Download Real Datasets

**For VinDr-Mammo (Training Data):**

1. Go to: https://physionet.org/content/vindr-mammo/
2. Complete credentialing process
3. Download dataset (~160 GB)
4. Extract to a directory
5. Update `config.py`:
   ```python
   VINDR_MAMMO_PATH = "/path/to/vindr_mammo"
   ```

**For INbreast (Test Data):**

1. Go to: http://medicalresearch.inescporto.pt/breastresearch/
2. Register and request access
3. Download dataset (~5 GB)
4. Extract to a directory
5. Update `config.py`:
   ```python
   INBREAST_PATH = "/path/to/inbreast"
   ```

**See detailed instructions:** `DATASET_SETUP_GUIDE.md`

### Option 3: Use Google Drive (For Colab)

1. **Upload datasets to Google Drive:**
   ```
   MyDrive/
   └── datasets/
       ├── vindr_mammo/
       │   ├── images/
       │   └── metadata.csv
       └── inbreast/
           ├── images/
           └── metadata.csv
   ```

2. **Mount in Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   VINDR_PATH = "/content/drive/MyDrive/datasets/vindr_mammo"
   INBREAST_PATH = "/content/drive/MyDrive/datasets/inbreast"
   ```

## Common Confusion

### "Where are the images?"

**Answer:** There are no actual images on GitHub. The repository contains:
- ✅ Code to load images
- ✅ Code to process images
- ✅ Code to train models on images
- ❌ Not the actual image files (too large)

### "Can I see example data?"

**Answer:** Yes! Run the Colab notebook:
1. Open: https://colab.research.google.com/github/dtobi59/mammography-multiobjective-optimization/blob/main/colab_tutorial.ipynb
2. Run Cell 5 (Create Demo Dataset)
3. It creates synthetic mammography images
4. You can inspect the demo data structure

### "How do I test without downloading 165 GB?"

**Answer:** Use the demo data in Colab! It creates small synthetic datasets that work with all the code.

## Verifying on GitHub

Go to: https://github.com/dtobi59/mammography-multiobjective-optimization

Click on these folders to see the Python code:

1. **data/** ← Click here, you'll see:
   - `__init__.py`
   - `augmentation.py`
   - `dataset.py`
   - `parsers.py`

2. **models/** ← Click here, you'll see:
   - `__init__.py`
   - `resnet.py`

3. **training/** ← Click here, you'll see:
   - `__init__.py`
   - `trainer.py`
   - `metrics.py`
   - `robustness.py`

All the code IS there! The actual image datasets are not (and shouldn't be).

## Summary

| Item | On GitHub? | Why? |
|------|-----------|------|
| `data/` Python module | ✅ Yes | Contains code for loading data |
| VinDr-Mammo images | ❌ No | Too large (160 GB) |
| INbreast images | ❌ No | Too large (5 GB) |
| Demo data | ❌ No | Created automatically in Colab |
| All other Python code | ✅ Yes | Part of the repository |
| Documentation | ✅ Yes | README, guides, etc. |
| Tests | ✅ Yes | All test files |
| Colab notebook | ✅ Yes | `colab_tutorial.ipynb` |

## What You Should Do

### For Quick Testing:
1. Open Colab notebook
2. Run all cells
3. Demo data created automatically
4. Everything works!

### For Real Research:
1. Download VinDr-Mammo and INbreast datasets
2. Place them in a local directory
3. Update paths in `config.py`
4. Run locally with real data

### For Colab with Real Data:
1. Upload datasets to Google Drive
2. Mount Drive in Colab
3. Point to your Drive folders
4. Run optimization

## Quick Test Right Now

Go to GitHub and click these links:

1. **View `data/` folder:**
   https://github.com/dtobi59/mammography-multiobjective-optimization/tree/main/data

2. **View `dataset.py` file:**
   https://github.com/dtobi59/mammography-multiobjective-optimization/blob/main/data/dataset.py

3. **View `parsers.py` file:**
   https://github.com/dtobi59/mammography-multiobjective-optimization/blob/main/data/parsers.py

You'll see all the code is there!

## Still Confused?

The `data/` directory on GitHub contains **Python code that loads and processes data**, not the actual medical images.

Think of it like:
- ✅ You have the **recipe** (code) on GitHub
- ❌ You don't have the **ingredients** (images) on GitHub
- 🎯 The **ingredients** must be downloaded or created separately

---

**Bottom line:** The code IS on GitHub. The actual datasets are NOT on GitHub (by design). Use the Colab demo data for testing, or download real datasets for production use.
