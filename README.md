# Multi-Objective Hyperparameter Optimization for Breast Cancer Classification

[![Tests](https://github.com/dtobi59/mammography-multiobjective-optimization/workflows/Tests/badge.svg)](https://github.com/dtobi59/mammography-multiobjective-optimization/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dtobi59/mammography-multiobjective-optimization/blob/main/colab_tutorial.ipynb)

Research-grade implementation of multi-objective hyperparameter optimization for CNN-based breast cancer classification under dataset shift.

**Author:** David ([@dtobi59](https://github.com/dtobi59))
**License:** MIT

---

## 🚀 Quick Start Options

### Option 1: Run on Google Colab (Recommended for Quick Start)

**No installation needed! Run with free GPU in your browser.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dtobi59/mammography-multiobjective-optimization/blob/main/colab_tutorial.ipynb)

Click the badge above and run all cells. The notebook includes:
- Automatic environment setup
- Demo dataset generation
- Complete optimization workflow
- Result visualization
- Ready in ~10 minutes!

See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed Colab instructions.

### Option 2: Run Locally

Clone and install on your local machine with GPU.

```bash
git clone https://github.com/dtobi59/mammography-multiobjective-optimization.git
cd mammography-multiobjective-optimization
pip install -r requirements.txt
```

See installation instructions below for details.

---

## Overview

This project implements a complete methodology for optimizing a ResNet-50 model for breast cancer classification using NSGA-III multi-objective optimization. The system:

- Trains on **VinDr-Mammo** (source domain)
- Evaluates zero-shot on **INbreast** (target domain)
- Optimizes 4 objectives simultaneously:
  1. Maximize PR-AUC
  2. Maximize AUROC
  3. Minimize Brier score
  4. Minimize robustness degradation

## Features

### Data Pipeline
- Patient-wise train/validation split (80/20)
- Image-level training with breast-level evaluation
- Noisy OR aggregation for combining CC and MLO views
- Intensity-only augmentation with strength parameter

### Model
- ResNet-50 with ImageNet pretrained weights
- Partial fine-tuning (controlled by hyperparameter)
- Binary classification with dropout

### Optimization
- 5 continuous hyperparameters optimized:
  - Learning rate (log scale)
  - Weight decay (log scale)
  - Dropout rate [0, 0.5]
  - Augmentation strength [0, 1]
  - Fraction of unfrozen backbone layers [0, 1]
- NSGA-III algorithm for many-objective optimization
- No surrogate models - each evaluation trains a full CNN
- Full reproducibility with fixed random seeds

### Evaluation
- Breast-level aggregation using Noisy OR
- Robustness evaluation under intensity perturbations
- Zero-shot transfer to INbreast
- Threshold transfer from source to target domain

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Interactive Tutorial

**📓 For a step-by-step interactive guide, see [tutorial.ipynb](tutorial.ipynb)**

The notebook covers:
- Dataset loading with dataset-specific parsers
- Model creation and training
- Robustness evaluation
- Noisy OR aggregation
- Zero-shot transfer to INbreast
- Pareto front analysis (after optimization)

### 1. Verify Setup

Before running optimization, verify your installation and data preparation:

```bash
python test_setup.py \
  --vindr_metadata path/to/vindr_mammo/metadata.csv \
  --vindr_images path/to/vindr_mammo/images \
  --inbreast_metadata path/to/inbreast/metadata.csv \
  --inbreast_images path/to/inbreast/images
```

This will check:
- ✓ All required dependencies are installed
- ✓ CUDA availability
- ✓ Data loading pipeline works
- ✓ Model creation works
- ✓ Metadata files are properly formatted
- ✓ Sample images can be loaded

### 2. Analyze Optimization Results

After optimization completes, analyze the Pareto front:

```bash
python optimization/analyze_pareto.py \
  --results_dir ./optimization_results \
  --output_dir ./pareto_analysis
```

This will:
- Print summary statistics of all Pareto solutions
- Identify extreme solutions (best for each objective)
- Generate 2D scatter plots of objective pairs
- Plot hyperparameter distributions
- Export configuration files for selected solutions

## Project Structure

```
project/
├── config.py                  # Configuration and hyperparameter bounds
├── test_setup.py              # Setup verification script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   ├── dataset.py            # Dataset classes with patient-wise splits
│   └── augmentation.py       # Intensity augmentation with strength parameter
├── models/
│   └── resnet.py             # ResNet-50 with partial fine-tuning
├── training/
│   ├── trainer.py            # Training loop with early stopping
│   ├── metrics.py            # Metric computation
│   └── robustness.py         # Robustness evaluation
├── optimization/
│   ├── problem.py            # pymoo Problem definition
│   ├── nsga3_runner.py       # NSGA-III optimization script
│   └── analyze_pareto.py     # Pareto front analysis and visualization
├── evaluation/
│   ├── evaluate_source.py    # Source validation evaluation
│   └── evaluate_target.py    # Zero-shot INbreast evaluation
└── utils/
    ├── seed.py               # Reproducibility utilities
    └── noisy_or.py           # Noisy OR aggregation
```

## Data Preparation

### Important: Dataset-Specific Structures

**VinDr-Mammo** and **INbreast have different directory structures and metadata formats**. This implementation provides dataset-specific parsers that convert each dataset to a unified internal representation.

**Key features:**
- ✅ Dataset-specific parsers for VinDr-Mammo and INbreast
- ✅ BI-RADS to binary label mapping (handles subcategories like 4A, 4B, 4C)
- ✅ Unified internal representation for all downstream code
- ✅ No dataset-specific logic beyond parsing

### Quick Setup

1. **VinDr-Mammo** (Source - training dataset):
   - Metadata CSV with columns: `image_id`, `study_id`, `laterality`, `view_position`, `breast_birads`
   - PNG images converted from DICOM
   - Configure in `config.py`: `VINDR_MAMMO_PATH` and `VINDR_CONFIG`

2. **INbreast** (Target - zero-shot evaluation only):
   - Metadata CSV or XML with patient ID, laterality, view, BI-RADS (with subcategories)
   - Different directory structure than VinDr-Mammo
   - Configure in `config.py`: `INBREAST_PATH` and `INBREAST_CONFIG`

### Detailed Setup Instructions

See **[DATASET_SETUP_GUIDE.md](DATASET_SETUP_GUIDE.md)** for:
- Complete metadata format specifications
- BI-RADS to binary label mapping rules
- Dataset-specific configuration examples
- Custom parser creation
- Troubleshooting common issues

### Unified Internal Representation

Both parsers output DataFrames with these standardized columns:
- `image_id`: Unique image identifier
- `patient_id`: Patient identifier (for patient-wise splitting)
- `breast_id`: Unique breast identifier (`patient_id` + laterality)
- `view`: View type ("CC" or "MLO")
- `label`: Binary label (0=benign, 1=malignant/suspicious)
- `image_path`: Relative path to image file
- `birads_original`: Original BI-RADS category (for reference)

**All training and evaluation code operates on this unified representation.**

## Configuration

Edit `config.py` to set your data paths and dataset-specific configurations:

```python
# Dataset paths
VINDR_MAMMO_PATH = "/path/to/vindr_mammo"
INBREAST_PATH = "/path/to/inbreast"

# VinDr-Mammo configuration (adjust column names to match your CSV)
VINDR_CONFIG = {
    "metadata_file": "metadata.csv",
    "image_dir": "images",
    "image_id_col": "image_id",
    "patient_id_col": "study_id",
    "laterality_col": "laterality",
    "view_col": "view_position",
    "birads_col": "breast_birads",
    "image_extension": ".png",
}

# INbreast configuration (adjust to match your metadata format)
INBREAST_CONFIG = {
    "metadata_file": "metadata.csv",
    "image_dir": "images",
    "metadata_format": "csv",  # or "xml"
    "patient_id_col": "patient_id",
    "laterality_col": "laterality",
    "view_col": "view",
    "birads_col": "birads",
    "filename_col": "file_name",
}
```

Additional configuration options:
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `BATCH_SIZE`: Training batch size (default: 16)
- `MAX_EPOCHS`: Maximum training epochs (default: 100)
- `EARLY_STOPPING_PATIENCE`: Early stopping patience (default: 15)
- `NSGA3_CONFIG`: NSGA-III population size and generations

See [DATASET_SETUP_GUIDE.md](DATASET_SETUP_GUIDE.md) for detailed configuration instructions.

## Usage

### 1. Run Multi-Objective Optimization

```bash
python optimization/nsga3_runner.py
```

This will:
- Load VinDr-Mammo metadata
- Create patient-wise train/validation split
- Run NSGA-III optimization
- Save Pareto front solutions to `./optimization_results/`

Output files:
- `pareto_solutions_[timestamp].csv`: Readable table of Pareto solutions
- `pareto_objectives_[timestamp].npy`: Objective values (4 x N array)
- `pareto_hyperparameters_[timestamp].npy`: Hyperparameter values (5 x N array)
- `metadata_[timestamp].json`: Optimization metadata
- `result_[timestamp].pkl`: Full pymoo Result object

#### Optimization Checkpoints

The optimization automatically saves checkpoints during execution:

```python
from optimization.nsga3_runner import NSGA3Runner

runner = NSGA3Runner(
    train_metadata=train_metadata,
    val_metadata=val_metadata,
    image_dir=image_dir,
    save_frequency=5  # Save every 5 generations
)

# Run optimization
result = runner.run()

# List all checkpoints
checkpoints = runner.list_checkpoints()

# Load a specific checkpoint
checkpoint = runner.load_checkpoint(checkpoints[0])

# Get Pareto front from checkpoint
pareto_df = runner.get_pareto_front_from_checkpoint(checkpoints[-1])
```

Checkpoint files are saved to `./optimization_results/optimization_checkpoints/`:
- `checkpoint_gen_XXXX.pkl`: Algorithm state for generation XXXX
- `pareto_gen_XXXX.csv`: Current Pareto front at generation XXXX

These checkpoints allow you to:
- Monitor optimization progress during long runs
- Analyze how the Pareto front evolves over generations
- Recover from interruptions (algorithm state is preserved)

### 2. Evaluate on Source Validation Set

```bash
python evaluation/evaluate_source.py \
  --checkpoint checkpoints/eval_X/best_checkpoint.pt \
  --hyperparameters hyperparameters.json \
  --metadata path/to/vindr_mammo/metadata.csv \
  --image_dir path/to/vindr_mammo/images
```

Example `hyperparameters.json`:
```json
{
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "dropout_rate": 0.2,
  "augmentation_strength": 0.5,
  "unfreeze_fraction": 0.3
}
```

This outputs:
- PR-AUC, AUROC, Brier score
- Robustness degradation
- Optimal threshold (selected via Youden's J statistic)
- Sensitivity and specificity at optimal threshold

### 3. Zero-Shot Evaluation on INbreast

```bash
python evaluation/evaluate_target.py \
  --checkpoint checkpoints/eval_X/best_checkpoint.pt \
  --hyperparameters hyperparameters.json \
  --threshold 0.45 \
  --metadata path/to/inbreast/metadata.csv \
  --image_dir path/to/inbreast/images
```

Where `--threshold` is the optimal threshold from source validation.

This outputs:
- PR-AUC, AUROC, Brier score on INbreast
- Sensitivity and specificity using transferred threshold
- **No fine-tuning or threshold tuning on target data**

## Methodology Details

### Objectives

All objectives are computed on the source validation set (VinDr-Mammo):

1. **PR-AUC** (Precision-Recall AUC): Breast-level performance
2. **AUROC** (ROC AUC): Breast-level discrimination
3. **Brier Score**: Calibration quality
4. **Robustness Degradation**: PR-AUC under standard inference minus PR-AUC under perturbed inference

Robustness perturbations:
- Fixed brightness adjustment (±10%)
- Fixed contrast scaling (±10%)
- Additive Gaussian noise (σ=0.02)

### Noisy OR Aggregation

For each breast with CC and MLO views:

```
p_breast = 1 - (1 - p_CC) * (1 - p_MLO)
```

If a view is missing, its probability is set to 0.

### Partial Fine-Tuning

The `unfreeze_fraction` hyperparameter controls which layers are trainable:
- `0.0`: Freeze entire backbone (only train classifier)
- `0.5`: Unfreeze last 50% of backbone layers
- `1.0`: Full fine-tuning

Layers are unfrozen from the end (deeper layers) backward.

### Early Stopping

Training uses early stopping based on validation PR-AUC:
- Patience: 15 epochs (configurable)
- Best checkpoint is automatically restored after training

## Reproducibility

All random seeds are fixed for reproducibility:
- Python `random` module
- NumPy
- PyTorch (including CUDA)
- Deterministic cuDNN operations

Set `RANDOM_SEED` in `config.py` to change the seed.

## Computational Requirements

- GPU with at least 8GB VRAM recommended
- Each CNN training takes ~1-2 hours (depends on dataset size)
- Full NSGA-III run: population_size × n_generations evaluations
  - Default: 24 × 50 = 1200 evaluations
  - Estimated time: ~50-100 days on single GPU
  - **Recommendation**: Use smaller population/generations for testing, or distribute across multiple GPUs

## Parallelization

The current implementation evaluates one configuration at a time. For faster optimization:

1. Modify `BreastCancerOptimizationProblem._evaluate()` to use parallel workers
2. Use multiple GPUs with `torch.nn.DataParallel` or `torch.distributed`
3. Reduce population size or number of generations in `config.py`

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@article{nsga3,
  title={An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach},
  author={Deb, Kalyanmoy and Jain, Himanshu},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2014}
}
```

## License

This implementation is provided for research purposes.

## Notes

- **No cross-validation**: Single fixed train/validation split
- **No threshold tuning on target**: Thresholds from source are transferred unchanged
- **No target data access during optimization**: Zero-shot evaluation only
- **Intensity-only augmentation**: No geometric transforms
- **Grayscale PNG inputs**: No DICOM windowing

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in `config.py`
- Use smaller image size (modify `IMAGE_SIZE` in `config.py`)
- Enable gradient checkpointing (requires model modification)

### Slow Training
- Reduce `MAX_EPOCHS` or `EARLY_STOPPING_PATIENCE`
- Use fewer dataloader workers (`num_workers` in dataset.py)
- Reduce population size or generations in NSGA-III

### Poor Performance
- Check data preprocessing (ensure correct normalization)
- Verify metadata format and labels
- Increase training epochs or reduce early stopping patience
- Try different hyperparameter bounds

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Ways to contribute:
- Report bugs or request features via [GitHub Issues](https://github.com/dtobi59/mammography-multiobjective-optimization/issues)
- Submit pull requests with improvements
- Add support for new datasets
- Improve documentation
- Share your results

## Citation

If you use this code in your research, please cite:

```bibtex
@software{david2026mammography,
  title={Multi-Objective Hyperparameter Optimization for Breast Cancer Classification under Dataset Shift},
  author={David},
  year={2026},
  url={https://github.com/dtobi59/mammography-multiobjective-optimization},
  version={1.0.0}
}
```

And cite NSGA-III:

```bibtex
@article{deb2014evolutionary,
  title={An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach},
  author={Deb, Kalyanmoy and Jain, Himanshu},
  journal={IEEE Transactions on Evolutionary Computation},
  volume={18},
  number={4},
  pages={577--601},
  year={2014},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VinDr-Mammo dataset providers
- INbreast dataset providers
- PyTorch and pymoo library developers
- ResNet architecture (He et al., 2016)

## Contact

- **GitHub Issues:** [Report bugs or request features](https://github.com/dtobi59/mammography-multiobjective-optimization/issues)
- **Author:** David ([@dtobi59](https://github.com/dtobi59))

---

**Repository:** https://github.com/dtobi59/mammography-multiobjective-optimization

**Status:** ✅ Production Ready | 114/114 Tests Passing
