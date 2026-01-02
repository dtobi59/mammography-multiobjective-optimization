"""
pymoo Problem definition for multi-objective hyperparameter optimization.
"""

import numpy as np
import torch
import pandas as pd
from pymoo.core.problem import Problem
from typing import Dict
import config
from utils.seed import set_all_seeds
from models import ResNet50WithPartialFineTuning
from data.dataset import create_dataloaders
from training.trainer import Trainer
from training.robustness import RobustnessEvaluator


class BreastCancerOptimizationProblem(Problem):
    """
    Multi-objective hyperparameter optimization problem for breast cancer classification.

    Objectives (all minimization):
    1. -PR-AUC (maximize PR-AUC)
    2. -AUROC (maximize AUROC)
    3. Brier score (minimize)
    4. Robustness degradation (minimize)

    Hyperparameters (continuous):
    1. Learning rate (log scale)
    2. Weight decay (log scale)
    3. Dropout rate [0, 0.5]
    4. Augmentation strength [0, 1]
    5. Unfreeze fraction [0, 1]
    """

    def __init__(
        self,
        train_metadata: pd.DataFrame,
        val_metadata: pd.DataFrame,
        image_dir: str,
        checkpoint_dir: str = "./checkpoints",
        n_workers: int = 4,
    ):
        """
        Initialize optimization problem.

        Args:
            train_metadata: Training metadata
            val_metadata: Validation metadata
            image_dir: Directory containing images
            checkpoint_dir: Directory to save model checkpoints
            n_workers: Number of dataloader workers
        """
        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.image_dir = image_dir
        self.checkpoint_dir = checkpoint_dir
        self.n_workers = n_workers

        # Define hyperparameter bounds
        # Variables: [log_lr, log_wd, dropout, aug_strength, unfreeze_frac]
        n_var = 5
        n_obj = 4  # 4 objectives

        # Lower bounds
        xl = np.array([
            np.log10(config.HYPERPARAMETER_BOUNDS["learning_rate"][0]),  # log_lr
            np.log10(config.HYPERPARAMETER_BOUNDS["weight_decay"][0]),   # log_wd
            config.HYPERPARAMETER_BOUNDS["dropout_rate"][0],              # dropout
            config.HYPERPARAMETER_BOUNDS["augmentation_strength"][0],     # aug_strength
            config.HYPERPARAMETER_BOUNDS["unfreeze_fraction"][0],         # unfreeze_frac
        ])

        # Upper bounds
        xu = np.array([
            np.log10(config.HYPERPARAMETER_BOUNDS["learning_rate"][1]),  # log_lr
            np.log10(config.HYPERPARAMETER_BOUNDS["weight_decay"][1]),   # log_wd
            config.HYPERPARAMETER_BOUNDS["dropout_rate"][1],              # dropout
            config.HYPERPARAMETER_BOUNDS["augmentation_strength"][1],     # aug_strength
            config.HYPERPARAMETER_BOUNDS["unfreeze_fraction"][1],         # unfreeze_frac
        ])

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=xl,
            xu=xu,
        )

        self.evaluation_counter = 0

    def _decode_hyperparameters(self, x: np.ndarray) -> Dict[str, float]:
        """
        Decode hyperparameters from optimization variables.

        Args:
            x: Optimization variables [log_lr, log_wd, dropout, aug_strength, unfreeze_frac]

        Returns:
            Dictionary of hyperparameters
        """
        return {
            "learning_rate": 10 ** x[0],      # Convert from log scale
            "weight_decay": 10 ** x[1],       # Convert from log scale
            "dropout_rate": x[2],
            "augmentation_strength": x[3],
            "unfreeze_fraction": x[4],
        }

    def _evaluate_single(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a single hyperparameter configuration.

        Args:
            x: Hyperparameter vector

        Returns:
            Objective values: [-PR-AUC, -AUROC, Brier, Robustness_degradation]
        """
        # Decode hyperparameters
        hparams = self._decode_hyperparameters(x)

        print(f"\n=== Evaluation {self.evaluation_counter + 1} ===")
        print(f"Hyperparameters: {hparams}")

        # Set random seeds for reproducibility
        set_all_seeds(config.RANDOM_SEED)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_metadata=self.train_metadata,
            val_metadata=self.val_metadata,
            image_dir=self.image_dir,
            batch_size=config.BATCH_SIZE,
            augmentation_strength=hparams["augmentation_strength"],
            num_workers=self.n_workers,
        )

        # Create model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ResNet50WithPartialFineTuning(
            unfreeze_fraction=hparams["unfreeze_fraction"],
            dropout_rate=hparams["dropout_rate"],
            pretrained=True,
        )

        # Train model
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_metadata=self.val_metadata,
            learning_rate=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            max_epochs=config.MAX_EPOCHS,
            device=device,
            checkpoint_dir=f"{self.checkpoint_dir}/eval_{self.evaluation_counter}",
        )

        best_metrics = trainer.train()

        # Evaluate robustness
        robustness_evaluator = RobustnessEvaluator(
            model=model,
            val_loader=val_loader,
            val_metadata=self.val_metadata,
            device=device,
        )
        robustness_degradation = robustness_evaluator.evaluate()

        # Compute objectives (all minimization)
        objectives = np.array([
            -best_metrics["pr_auc"],         # Maximize PR-AUC
            -best_metrics["auroc"],          # Maximize AUROC
            best_metrics["brier"],           # Minimize Brier score
            robustness_degradation,          # Minimize robustness degradation
        ])

        print(f"Objectives: PR-AUC={best_metrics['pr_auc']:.4f}, "
              f"AUROC={best_metrics['auroc']:.4f}, "
              f"Brier={best_metrics['brier']:.4f}, "
              f"Robustness={robustness_degradation:.4f}")

        self.evaluation_counter += 1

        return objectives

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate multiple hyperparameter configurations.

        Args:
            x: Array of hyperparameter vectors, shape (n_samples, n_var)
            out: Output dictionary to store objectives
        """
        # Evaluate each configuration
        objectives = np.array([self._evaluate_single(xi) for xi in x])

        # Store in output dictionary
        out["F"] = objectives
