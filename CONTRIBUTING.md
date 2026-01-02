# Contributing to Mammography Multi-Objective Optimization

Thank you for your interest in contributing! This project implements research-grade code for multi-objective hyperparameter optimization of CNNs for breast cancer classification.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, PyTorch version)
- Error messages and stack traces

### Suggesting Enhancements

We welcome suggestions for:
- New dataset parsers (DDSM, CBIS-DDSM, etc.)
- Additional objectives or metrics
- Performance improvements
- Documentation improvements

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/dtobi59/mammography-multiobjective-optimization.git
   cd mammography-multiobjective-optimization
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow PEP 8 style guide
   - Add docstrings to functions
   - Update documentation if needed
   - Add tests for new functionality

4. **Run tests**
   ```bash
   python test_correctness.py
   python test_parsers.py
   python test_integration.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/dtobi59/mammography-multiobjective-optimization.git
cd mammography-multiobjective-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_correctness.py
```

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all functions (Google style)
- Keep functions focused and modular
- Add comments for complex logic

Example:
```python
def compute_metric(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute evaluation metric.

    Args:
        predictions: Predicted probabilities, shape (n_samples,)
        labels: Ground truth labels, shape (n_samples,)

    Returns:
        Metric value in [0, 1]
    """
    # Implementation
    return metric_value
```

## Adding New Datasets

To add support for a new dataset:

1. Create a parser in `data/parsers.py`
2. Inherit from existing parsers or implement from scratch
3. Output standardized DataFrame format
4. Add configuration to `config.py`
5. Add tests in `test_parsers.py`
6. Update `DATASET_SETUP_GUIDE.md`

Example:
```python
class NewDatasetParser:
    def parse(self) -> pd.DataFrame:
        # Parse dataset-specific format
        # Return standardized DataFrame
        pass
```

## Testing

All contributions must pass existing tests and include new tests for new features:

```bash
# Run all tests
python test_correctness.py  # 79 tests
python test_parsers.py      # 34 tests
python test_integration.py  # 1 test
```

Test coverage requirements:
- New parsers: 100% coverage
- New metrics: Test edge cases
- New features: Integration test

## Documentation

Update documentation for any changes:
- `README.md` - User-facing documentation
- `DATASET_SETUP_GUIDE.md` - Dataset-specific instructions
- `IMPLEMENTATION_NOTES.md` - Technical details
- Docstrings - In-code documentation

## Research Reproducibility

This is research code. Contributions must maintain:
- ✅ Deterministic behavior (fixed seeds)
- ✅ Exact specification compliance
- ✅ No approximations or shortcuts
- ✅ Comprehensive testing

## Questions?

- Open an issue for questions
- Check existing documentation first
- Review test files for usage examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Contributors will be listed in the README. Thank you for helping improve this project!
