# Implementation Notes

This document provides detailed technical notes about the implementation of the multi-objective hyperparameter optimization system for breast cancer classification.

## Implementation Fidelity to Specification

This implementation strictly follows the provided specification without simplification or invention of methods. Key points:

### Data Handling
- ✓ Source dataset: VinDr-Mammo (PNG images)
- ✓ Target dataset: INbreast (zero-shot evaluation only)
- ✓ Patient-wise 80/20 train/validation split
- ✓ Single fixed split, no cross-validation
- ✓ Image-level training, breast-level evaluation

### Model Architecture
- ✓ ResNet-50 with ImageNet pretrained weights
- ✓ Binary classification with sigmoid output
- ✓ Partial fine-tuning controlled by hyperparameter (fraction of unfrozen layers)
- ✓ Dropout in classification head

### Training Configuration
- ✓ AdamW optimizer (fixed, not optimized)
- ✓ Binary cross-entropy loss
- ✓ Fixed batch size
- ✓ Early stopping on validation PR-AUC
- ✓ Best checkpoint restoration
- ✓ Fixed random seeds for reproducibility

### Hyperparameters Optimized (5 continuous)
- ✓ Learning rate (log scale)
- ✓ Weight decay (log scale)
- ✓ Dropout rate [0, 0.5]
- ✓ Augmentation strength [0, 1]
- ✓ Fraction of unfrozen backbone layers [0, 1]

### Augmentation
- ✓ Scalar strength parameter linearly scales magnitude
- ✓ Intensity-only: brightness, contrast, Gaussian noise
- ✓ No geometric transforms
- ✓ Training only (not applied during validation)

### Objectives (4, all on validation set)
- ✓ Maximize PR-AUC → minimize -PR-AUC
- ✓ Maximize AUROC → minimize -AUROC
- ✓ Minimize Brier score
- ✓ Minimize robustness degradation: R = PR-AUC_standard - PR-AUC_perturbed

### Robustness Evaluation
- ✓ Mild intensity perturbations at inference time
- ✓ Fixed perturbations (brightness, contrast, noise)
- ✓ Compare standard vs perturbed PR-AUC

### Optimization
- ✓ NSGA-III from pymoo
- ✓ Many-objective (4 objectives)
- ✓ Reference directions
- ✓ Fixed population size and generations
- ✓ No surrogate models
- ✓ Each evaluation trains full CNN

### Evaluation Protocol
- ✓ Pareto front selection
- ✓ Zero-shot on INbreast (no fine-tuning, no threshold tuning)
- ✓ Same preprocessing and Noisy OR aggregation
- ✓ Thresholds selected on source validation and transferred

## Technical Implementation Details

### 1. Noisy OR Aggregation

**Formula:** `p_breast = 1 - (1 - p_CC) * (1 - p_MLO)`

**Implementation:** `utils/noisy_or.py:noisy_or_aggregation()`

**Edge cases handled:**
- Missing views: probability set to 0.0
- Multiple views of same type: take max probability (most suspicious)

### 2. Partial Fine-Tuning

**Strategy:** Unfreeze layers from end (deeper layers) backward

**Implementation:** `models/resnet.py:_setup_partial_finetuning()`

**Examples:**
- `unfreeze_fraction=0.0`: Freeze all backbone, train only classifier
- `unfreeze_fraction=0.3`: Unfreeze last 30% of layers
- `unfreeze_fraction=1.0`: Full fine-tuning

### 3. Augmentation Strength

**Implementation:** `data/augmentation.py:IntensityAugmentation`

**Scaling:**
- Brightness: range = `[-strength * 0.2, +strength * 0.2]`
- Contrast: range = `[-strength * 0.2, +strength * 0.2]`
- Noise std: `strength * 0.05`

**strength=0.0:** No augmentation
**strength=1.0:** Full augmentation

### 4. Early Stopping

**Metric:** Validation PR-AUC (breast-level)
**Patience:** 15 epochs (configurable)
**Mode:** Maximize

**Implementation:** `training/trainer.py:EarlyStopping`

### 5. Robustness Perturbation

**Fixed perturbations (not random):**
- Brightness: +10%
- Contrast: +10%
- Noise std: 0.02

**Implementation:** `data/augmentation.py:RobustnessPerturbation`

### 6. Decision Threshold Selection

**Method:** Youden's J statistic

**Formula:** `J = sensitivity + specificity - 1`

**Implementation:** `training/metrics.py:find_optimal_threshold()`

Searches over 101 thresholds in [0, 1] to find maximum J.

### 7. NSGA-III Configuration

**Reference directions:** "energy" method from pymoo
**Population initialization:** Latin hypercube sampling (pymoo default)
**Termination:** Fixed number of generations

**Implementation:** `optimization/nsga3_runner.py`

## File Organization

### Configuration (`config.py`)
- All hyperparameter bounds
- Fixed training parameters
- Data paths (user-configurable)
- NSGA-III settings

### Data Pipeline (`data/`)
- `dataset.py`: PyTorch Dataset, patient-wise splitting
- `augmentation.py`: Intensity augmentation and robustness perturbation

### Model (`models/`)
- `resnet.py`: ResNet-50 with partial fine-tuning and dropout

### Training (`training/`)
- `trainer.py`: Training loop with early stopping
- `metrics.py`: PR-AUC, AUROC, Brier, sensitivity, specificity
- `robustness.py`: Robustness evaluation under perturbations

### Optimization (`optimization/`)
- `problem.py`: pymoo Problem class, encapsulates full training pipeline
- `nsga3_runner.py`: NSGA-III runner with result logging
- `analyze_pareto.py`: Pareto front analysis and visualization

### Evaluation (`evaluation/`)
- `evaluate_source.py`: Source validation metrics + threshold selection
- `evaluate_target.py`: Zero-shot INbreast evaluation with transferred threshold

### Utilities (`utils/`)
- `seed.py`: Reproducibility (fix all random seeds)
- `noisy_or.py`: Breast-level aggregation

## Reproducibility

All randomness is controlled:

1. **Python random module:** `random.seed(RANDOM_SEED)`
2. **NumPy:** `np.random.seed(RANDOM_SEED)`
3. **PyTorch CPU:** `torch.manual_seed(RANDOM_SEED)`
4. **PyTorch CUDA:** `torch.cuda.manual_seed_all(RANDOM_SEED)`
5. **cuDNN:** `torch.backends.cudnn.deterministic = True`

Set `RANDOM_SEED` in `config.py` to change seed globally.

## Computational Considerations

### Training Time per Configuration
- Depends on dataset size and hardware
- Typical: 1-2 hours per configuration on modern GPU
- Early stopping can terminate earlier

### Full NSGA-III Run
- Population size: 24 (default)
- Generations: 50 (default)
- Total evaluations: 24 × 50 = 1,200
- Estimated time: ~50-100 days on single GPU

### Recommendations for Faster Experimentation
1. **Reduce population/generations:** Edit `NSGA3_CONFIG` in `config.py`
   - Start with pop_size=8, n_generations=10 for testing
2. **Parallelize evaluations:** Modify `problem.py` to evaluate multiple configurations in parallel
3. **Use multiple GPUs:** Implement distributed training
4. **Reduce MAX_EPOCHS:** Lower maximum training epochs in `config.py`

## Extension Points

### Adding New Objectives
1. Compute new metric in `training/metrics.py`
2. Add to `BreastCancerOptimizationProblem._evaluate_single()` in `optimization/problem.py`
3. Update `n_obj` in Problem initialization
4. Update `NSGA3_CONFIG["n_objectives"]` in `config.py`

### Adding New Hyperparameters
1. Add bounds to `HYPERPARAMETER_BOUNDS` in `config.py`
2. Update `_decode_hyperparameters()` in `optimization/problem.py`
3. Update `n_var` in Problem initialization
4. Use hyperparameter in model/training pipeline

### Using Different Backbone
1. Modify `models/resnet.py` to use different architecture
2. Adjust `IMAGE_SIZE` in `config.py` if needed
3. Update feature dimension in classification head

### Custom Augmentation
1. Modify `data/augmentation.py:IntensityAugmentation`
2. Keep strength parameter for hyperparameter optimization
3. Ensure augmentation is intensity-only (per specification)

## Known Limitations

1. **No multi-GPU support:** Current implementation uses single GPU
2. **Sequential evaluation:** Evaluates one configuration at a time
3. **Fixed image size:** 224×224 (ResNet-50 default)
4. **No data caching:** Images loaded from disk each time
5. **Fixed augmentation types:** Only brightness, contrast, noise

## Testing

Use `test_setup.py` to verify:
- Dependencies installed
- CUDA available
- Data properly formatted
- Metadata has required columns
- Images loadable
- Model creation works
- Data pipeline works

## Metrics Definitions

### PR-AUC (Precision-Recall AUC)
- Area under precision-recall curve
- Better for imbalanced datasets
- Range: [0, 1], higher is better

### AUROC (ROC AUC)
- Area under receiver operating characteristic curve
- Discrimination ability
- Range: [0, 1], higher is better (0.5 = random)

### Brier Score
- Mean squared error of probabilistic predictions
- Calibration quality
- Range: [0, 1], lower is better

### Robustness Degradation
- `R = PR-AUC_standard - PR-AUC_perturbed`
- Measures performance drop under perturbations
- Can be negative (perturbed better than standard)
- Lower is better (more robust)

### Sensitivity (Recall, True Positive Rate)
- `TP / (TP + FN)`
- Proportion of actual positives correctly identified

### Specificity (True Negative Rate)
- `TN / (TN + FP)`
- Proportion of actual negatives correctly identified

### Youden's J Statistic
- `J = sensitivity + specificity - 1`
- Range: [0, 1]
- Maximizing J balances sensitivity and specificity

## Data Format Requirements

### Metadata CSV Columns
1. **image_id** (string): Unique identifier for each image
2. **patient_id** (string): Patient identifier
3. **breast_id** (string): Unique identifier for each breast
4. **view** (string): "CC" or "MLO"
5. **label** (int): 0 (benign) or 1 (malignant)
6. **image_path** (string): Relative path from image_dir

### Image Format
- **Format:** PNG (grayscale or RGB)
- **Bit depth:** Any (will be normalized to [0, 1])
- **Size:** Any (will be resized to 224×224)
- **Channels:** Grayscale preferred (RGB will work)

## Troubleshooting Common Issues

### 1. Out of Memory
**Symptoms:** CUDA out of memory error
**Solutions:**
- Reduce `BATCH_SIZE` in `config.py`
- Use smaller `IMAGE_SIZE`
- Reduce `unfreeze_fraction` (fewer trainable parameters)

### 2. Slow Training
**Symptoms:** Training takes too long
**Solutions:**
- Reduce `MAX_EPOCHS`
- Reduce `EARLY_STOPPING_PATIENCE`
- Use fewer dataloader workers
- Reduce image size

### 3. Poor Convergence
**Symptoms:** Validation metrics not improving
**Solutions:**
- Increase `MAX_EPOCHS`
- Widen hyperparameter search bounds
- Check data preprocessing
- Verify labels are correct

### 4. Pareto Front Too Small
**Symptoms:** Few non-dominated solutions
**Solutions:**
- Increase population size
- Increase number of generations
- Verify objectives are conflicting

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{breast_cancer_moho,
  title={Multi-Objective Hyperparameter Optimization for Breast Cancer Classification},
  author={[Your Name]},
  year={2024},
  url={[Your Repository URL]}
}
```

Also cite NSGA-III:

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
