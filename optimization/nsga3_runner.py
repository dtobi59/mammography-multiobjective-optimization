"""
NSGA-III optimization runner with logging.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.result import Result
import config
from .problem import BreastCancerOptimizationProblem


class NSGA3Runner:
    """
    Runner for NSGA-III multi-objective optimization.
    """

    def __init__(
        self,
        train_metadata: pd.DataFrame,
        val_metadata: pd.DataFrame,
        image_dir: str,
        output_dir: str = "./optimization_results",
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Initialize NSGA-III runner.

        Args:
            train_metadata: Training metadata
            val_metadata: Validation metadata
            image_dir: Directory containing images
            output_dir: Directory to save optimization results
            checkpoint_dir: Directory to save model checkpoints
        """
        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create problem
        self.problem = BreastCancerOptimizationProblem(
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            image_dir=image_dir,
            checkpoint_dir=checkpoint_dir,
        )

    def run(self) -> Result:
        """
        Run NSGA-III optimization.

        Returns:
            Optimization result
        """
        # Create reference directions for NSGA-III
        # For 4 objectives, use Das-Dennis approach
        ref_dirs = get_reference_directions(
            "energy",
            n_dim=config.NSGA3_CONFIG["n_objectives"],
            n_points=config.NSGA3_CONFIG["pop_size"],
        )

        # Initialize NSGA-III algorithm
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=config.NSGA3_CONFIG["pop_size"],
        )

        # Run optimization
        print("=" * 80)
        print("Starting NSGA-III optimization")
        print(f"Population size: {config.NSGA3_CONFIG['pop_size']}")
        print(f"Generations: {config.NSGA3_CONFIG['n_generations']}")
        print(f"Objectives: {config.NSGA3_CONFIG['n_objectives']}")
        print("=" * 80)

        start_time = datetime.now()

        result = minimize(
            self.problem,
            algorithm,
            termination=('n_gen', config.NSGA3_CONFIG['n_generations']),
            seed=config.RANDOM_SEED,
            verbose=True,
            save_history=True,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("=" * 80)
        print(f"Optimization completed in {duration:.2f} seconds")
        print(f"Number of non-dominated solutions: {len(result.F)}")
        print("=" * 80)

        # Save results
        self._save_results(result, duration)

        return result

    def _save_results(self, result: Result, duration: float) -> None:
        """
        Save optimization results.

        Args:
            result: Optimization result
            duration: Optimization duration in seconds
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save Pareto front (objectives)
        pareto_objectives = result.F
        np.save(
            os.path.join(self.output_dir, f"pareto_objectives_{timestamp}.npy"),
            pareto_objectives
        )

        # Save Pareto set (hyperparameters)
        pareto_hyperparameters = result.X
        np.save(
            os.path.join(self.output_dir, f"pareto_hyperparameters_{timestamp}.npy"),
            pareto_hyperparameters
        )

        # Convert to readable format
        pareto_df = self._create_pareto_dataframe(result)
        pareto_df.to_csv(
            os.path.join(self.output_dir, f"pareto_solutions_{timestamp}.csv"),
            index=False
        )

        # Save optimization metadata
        metadata = {
            "timestamp": timestamp,
            "duration_seconds": duration,
            "n_solutions": len(result.F),
            "pop_size": config.NSGA3_CONFIG["pop_size"],
            "n_generations": config.NSGA3_CONFIG["n_generations"],
            "n_objectives": config.NSGA3_CONFIG["n_objectives"],
            "random_seed": config.RANDOM_SEED,
        }

        with open(os.path.join(self.output_dir, f"metadata_{timestamp}.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save full result object
        with open(os.path.join(self.output_dir, f"result_{timestamp}.pkl"), "wb") as f:
            pickle.dump(result, f)

        print(f"Results saved to {self.output_dir}")

    def _create_pareto_dataframe(self, result: Result) -> pd.DataFrame:
        """
        Create a readable DataFrame of Pareto solutions.

        Args:
            result: Optimization result

        Returns:
            DataFrame with hyperparameters and objectives
        """
        rows = []

        for i, (x, f) in enumerate(zip(result.X, result.F)):
            # Decode hyperparameters
            hparams = self.problem._decode_hyperparameters(x)

            # Create row
            row = {
                "solution_id": i,
                "learning_rate": hparams["learning_rate"],
                "weight_decay": hparams["weight_decay"],
                "dropout_rate": hparams["dropout_rate"],
                "augmentation_strength": hparams["augmentation_strength"],
                "unfreeze_fraction": hparams["unfreeze_fraction"],
                "pr_auc": -f[0],  # Convert back to maximization
                "auroc": -f[1],   # Convert back to maximization
                "brier": f[2],
                "robustness_degradation": f[3],
            }

            rows.append(row)

        return pd.DataFrame(rows)


def load_metadata(
    dataset_name: str,
    dataset_path: str,
    dataset_config: dict
) -> pd.DataFrame:
    """
    Load and parse dataset metadata using dataset-specific parser.

    Args:
        dataset_name: Dataset name ('vindr' or 'inbreast')
        dataset_path: Base path to dataset
        dataset_config: Dataset configuration dictionary

    Returns:
        Standardized metadata DataFrame with columns:
        - image_id, patient_id, breast_id, view, label, image_path, birads_original
    """
    from pathlib import Path
    from data.parsers import parse_dataset

    metadata_path = str(Path(dataset_path) / dataset_config["metadata_file"])
    image_dir = str(Path(dataset_path) / dataset_config["image_dir"])

    # Extract parser kwargs from config (exclude metadata_file and image_dir)
    parser_kwargs = {
        k: v for k, v in dataset_config.items()
        if k not in ["metadata_file", "image_dir"]
    }

    metadata = parse_dataset(
        dataset_name=dataset_name,
        metadata_path=metadata_path,
        image_dir=image_dir,
        **parser_kwargs
    )

    # Validate required columns
    required_columns = ["image_id", "patient_id", "breast_id", "view", "label", "image_path"]
    missing_columns = set(required_columns) - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in metadata: {missing_columns}")

    return metadata


if __name__ == "__main__":
    """
    Example usage:

    python optimization/nsga3_runner.py
    """
    import sys
    from pathlib import Path

    # Load VinDr-Mammo metadata using dataset-specific parser
    print("Loading VinDr-Mammo metadata...")
    vindr_metadata = load_metadata(
        dataset_name="vindr",
        dataset_path=config.VINDR_MAMMO_PATH,
        dataset_config=config.VINDR_CONFIG
    )

    # Create train/val split (patient-wise)
    from data.dataset import create_train_val_split
    train_metadata, val_metadata = create_train_val_split(vindr_metadata)

    print(f"Train samples: {len(train_metadata)}")
    print(f"Validation samples: {len(val_metadata)}")
    print(f"Unique patients - Train: {train_metadata['patient_id'].nunique()}, "
          f"Val: {val_metadata['patient_id'].nunique()}")

    # Create runner
    image_dir = str(Path(config.VINDR_MAMMO_PATH) / config.VINDR_CONFIG["image_dir"])
    runner = NSGA3Runner(
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        image_dir=image_dir,
    )

    # Run optimization
    result = runner.run()

    print("\nOptimization complete!")
    print(f"Pareto front size: {len(result.F)}")
    print(f"\nResults saved to: {runner.output_dir}")
