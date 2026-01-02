# Running on Google Colab

Complete guide to running this project on Google Colab with GPU acceleration.

## Quick Start

### Method 1: One-Click Launch (Recommended)

Click this badge to open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dtobi59/mammography-multiobjective-optimization/blob/main/colab_tutorial.ipynb)

Then just run all cells from top to bottom!

### Method 2: Manual Setup

1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub
3. Enter: `dtobi59/mammography-multiobjective-optimization`
4. Select: `colab_tutorial.ipynb`
5. Run all cells

## Prerequisites

### 1. Google Account
You need a Google account to use Colab.

### 2. GPU Runtime (Recommended)
Enable GPU for faster training:
- Runtime → Change runtime type
- Hardware accelerator → GPU (T4)
- Save

### 3. Datasets

You have three options:

#### Option A: Use Demo Dataset (Easiest)
The notebook creates a small synthetic dataset automatically.
- Best for: Testing the pipeline
- Setup time: Instant
- Dataset size: ~20 images

#### Option B: Upload to Colab (Small Datasets)
Upload your data files directly to Colab session.
- Best for: Small datasets (<500 MB)
- Setup time: 5-10 minutes
- Note: Files deleted when session ends

#### Option C: Use Google Drive (Recommended for Real Data)
Store datasets in Google Drive and mount in Colab.
- Best for: Large datasets (>500 MB)
- Setup time: 15-30 minutes
- Persistent storage

## Google Drive Setup (Option C)

### Step 1: Prepare Your Data

Organize your data in Google Drive:

```
MyDrive/
└── datasets/
    ├── vindr_mammo/
    │   ├── images/
    │   │   ├── image1.png
    │   │   ├── image2.png
    │   │   └── ...
    │   └── metadata.csv
    └── inbreast/
        ├── images/
        │   ├── image1.png
        │   ├── image2.png
        │   └── ...
        └── metadata.csv
```

### Step 2: Upload Data to Google Drive

**For large datasets:**

1. **Download datasets** to your computer:
   - VinDr-Mammo: https://physionet.org/content/vindr-mammo/
   - INbreast: http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database

2. **Upload to Google Drive:**
   - Open Google Drive in browser
   - Create folder: `datasets`
   - Drag and drop dataset folders
   - Wait for upload to complete (may take hours for large datasets)

**Tip:** Use [Google Drive Desktop](https://www.google.com/drive/download/) for faster upload of large files.

### Step 3: Mount in Colab

In the notebook, uncomment and run:

```python
from google.colab import drive
drive.mount('/content/drive')

# Set paths
VINDR_PATH = "/content/drive/MyDrive/datasets/vindr_mammo"
INBREAST_PATH = "/content/drive/MyDrive/datasets/inbreast"
```

## Configuration

### For Quick Testing (Demo)

The notebook is pre-configured for quick testing:
- Population size: 6 (default: 20)
- Generations: 5 (default: 100)
- Max epochs: 3 (default: 100)

**Expected runtime:** ~10-15 minutes on GPU

### For Production Runs

Edit the configuration cell:

```python
# In the "Configure Paths" cell, change these lines:

config_content = config_content.replace(
    '\"pop_size\": 6,  # Reduced for demo',
    '\"pop_size\": 20,'  # Production
)
config_content = config_content.replace(
    '\"n_generations\": 5,  # Reduced for demo',
    '\"n_generations\": 100,'  # Production
)
config_content = config_content.replace(
    'MAX_EPOCHS = 3  # Reduced for demo',
    'MAX_EPOCHS = 100'  # Production
)
```

**Expected runtime:** Several hours to days depending on dataset size

## Running the Notebook

### Step-by-Step Guide

1. **Setup Environment** (Cell 1-3)
   - Checks GPU availability
   - Clones repository from GitHub
   - Installs dependencies
   - Time: ~2 minutes

2. **Dataset Setup** (Cell 4-6)
   - Choose one option (A, B, or C)
   - Creates or loads data
   - Time: Instant (demo) to 30 min (upload)

3. **Configure Paths** (Cell 7)
   - Updates config.py with your paths
   - Time: <1 second

4. **Verify Setup** (Cell 8)
   - Tests data loading
   - Shows dataset statistics
   - Time: ~10 seconds

5. **Run Optimization** (Cell 9)
   - Runs NSGA-III algorithm
   - Saves checkpoints automatically
   - Time: 10 min (demo) to hours (production)

6. **Inspect Checkpoints** (Cell 10)
   - Lists saved checkpoints
   - Shows progress over generations
   - Time: ~5 seconds

7. **Analyze Results** (Cell 11-12)
   - Loads Pareto front
   - Creates visualizations
   - Identifies best solutions
   - Time: ~30 seconds

8. **Download Results** (Cell 13)
   - Creates zip file
   - Downloads to your computer
   - Time: ~1 minute

## Tips for Colab

### 1. Session Management

**Colab sessions timeout after:**
- 12 hours (free tier)
- 24 hours (Colab Pro)

**To prevent timeouts:**
- Keep browser tab open
- Run this in a cell to prevent disconnect:
  ```python
  # Keep session alive
  import time
  from IPython.display import display, Javascript

  display(Javascript('''
    function KeepClicking(){
      console.log("Clicking");
      document.querySelector("colab-toolbar-button#connect").click()
    }
    setInterval(KeepClicking, 60000)
  '''))
  ```

### 2. Save Results Frequently

**Important:** Colab sessions are temporary!

Save results to Google Drive:
```python
# After optimization completes
!cp -r optimization_results /content/drive/MyDrive/
!cp -r checkpoints /content/drive/MyDrive/
```

### 3. Monitor Progress

Check checkpoints while optimization runs:
```python
# Run in a separate cell while optimization is running
checkpoints = runner.list_checkpoints()
print(f"Completed {len(checkpoints)} generations")

# View latest Pareto front
if checkpoints:
    latest = runner.get_pareto_front_from_checkpoint(checkpoints[-1])
    print(latest)
```

### 4. GPU Usage

Check GPU memory:
```python
!nvidia-smi
```

If you get "CUDA out of memory":
- Reduce batch size in config.py
- Reduce population size
- Restart runtime: Runtime → Restart runtime

### 5. Resuming Interrupted Runs

If your session disconnects:

1. **Re-run setup cells** (1-8)
2. **Check for existing checkpoints:**
   ```python
   checkpoints = runner.list_checkpoints()
   print(f"Found {len(checkpoints)} checkpoints")
   ```
3. **Continue or restart:**
   - If checkpoints exist, you can analyze partial results
   - To restart: delete checkpoint directory and re-run

## Common Issues

### Issue 1: "No module named 'XXX'"

**Solution:** Re-run the dependency installation cell:
```bash
!pip install -q -r requirements.txt
```

### Issue 2: "CUDA out of memory"

**Solution:**
- Runtime → Factory reset runtime
- Reduce batch size in config.py (line ~30):
  ```python
  BATCH_SIZE = 8  # Reduce from 16
  ```

### Issue 3: "FileNotFoundError: Image not found"

**Solution:**
- Check dataset paths are correct
- Verify files exist in Google Drive
- Re-mount Google Drive

### Issue 4: Session disconnected during optimization

**Solution:**
- Checkpoints are saved every generation
- Re-run setup cells
- Check checkpoints to see progress
- Optimization results are still usable if >1 generation completed

### Issue 5: Slow training

**Solution:**
- Verify GPU is enabled: Runtime → Change runtime type → GPU
- Check GPU usage: `!nvidia-smi`
- If no GPU available, try reconnecting or switching to Colab Pro

## Downloading Results

### Method 1: Files Panel (Small Files)

1. Click folder icon on left sidebar
2. Navigate to `optimization_results/`
3. Right-click file → Download

### Method 2: Zip and Download (Large Files)

```python
!zip -r results.zip optimization_results/ checkpoints/
from google.colab import files
files.download('results.zip')
```

### Method 3: Save to Google Drive (Best)

```python
# Copy to Google Drive
!cp -r optimization_results /content/drive/MyDrive/breast_cancer_results/
!cp -r checkpoints /content/drive/MyDrive/breast_cancer_results/

print("Results saved to Google Drive!")
```

## Performance Expectations

### Demo Dataset (Default Settings)
- Population: 6
- Generations: 5
- Epochs: 3
- Dataset: 20 synthetic images
- **Time:** ~10-15 minutes on GPU

### Small Real Dataset
- Population: 10
- Generations: 20
- Epochs: 50
- Dataset: ~500 images
- **Time:** ~3-4 hours on GPU

### Full Production Run
- Population: 20
- Generations: 100
- Epochs: 100
- Dataset: ~5,000 images
- **Time:** ~2-3 days on GPU
- **Recommendation:** Use Colab Pro or split into smaller runs

## Cost

### Google Colab Free
- ✅ Free GPU access (T4)
- ⚠️ Session limits: 12 hours
- ⚠️ May get disconnected during peak times
- Best for: Testing and small runs

### Google Colab Pro ($10/month)
- ✅ Better GPUs (A100, V100 available)
- ✅ Longer sessions: 24 hours
- ✅ Priority access
- Best for: Production runs

### Google Colab Pro+ ($50/month)
- ✅ Best GPUs (A100 guaranteed)
- ✅ Longest sessions: up to background execution
- ✅ Highest priority
- Best for: Large-scale experiments

## Next Steps After Colab

1. **Download trained models**
   - Best Pareto solutions saved as checkpoints
   - Use for deployment or further analysis

2. **Evaluate on your own data**
   - Upload new test set
   - Run evaluation scripts

3. **Run locally**
   - Clone repository to local machine
   - Use saved checkpoints from Colab
   - Continue experimentation

4. **Deploy**
   - Export best model
   - Create inference pipeline
   - Deploy to production

## Resources

- **Colab Documentation:** https://colab.research.google.com/
- **Colab FAQ:** https://research.google.com/colaboratory/faq.html
- **GPU Guide:** https://colab.research.google.com/notebooks/gpu.ipynb
- **Repository:** https://github.com/dtobi59/mammography-multiobjective-optimization

## Support

If you encounter issues:

1. Check this guide's Common Issues section
2. Review Colab's official documentation
3. Open an issue on GitHub: https://github.com/dtobi59/mammography-multiobjective-optimization/issues

---

**Happy optimizing! 🚀**
