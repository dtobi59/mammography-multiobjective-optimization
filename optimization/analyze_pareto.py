"""
Analyze and visualize Pareto front from NSGA-III optimization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import argparse


def load_pareto_solutions(results_dir: str, timestamp: Optional[str] = None) -> pd.DataFrame:
    """
    Load Pareto solutions from CSV.

    Args:
        results_dir: Directory containing optimization results
        timestamp: Optional timestamp of results file. If None, loads the most recent.

    Returns:
        DataFrame of Pareto solutions
    """
    if timestamp is not None:
        csv_path = os.path.join(results_dir, f"pareto_solutions_{timestamp}.csv")
    else:
        # Find most recent results file
        csv_files = [f for f in os.listdir(results_dir) if f.startswith("pareto_solutions_")]
        if not csv_files:
            raise FileNotFoundError(f"No Pareto solution files found in {results_dir}")
        csv_files.sort(reverse=True)  # Most recent first
        csv_path = os.path.join(results_dir, csv_files[0])

    print(f"Loading Pareto solutions from: {csv_path}")
    return pd.read_csv(csv_path)


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics of Pareto front."""
    print("\n" + "=" * 80)
    print("PARETO FRONT SUMMARY")
    print("=" * 80)
    print(f"Number of solutions: {len(df)}")
    print("\nObjective Statistics:")
    print("-" * 80)

    objectives = ["pr_auc", "auroc", "brier", "robustness_degradation"]
    for obj in objectives:
        if obj in df.columns:
            print(f"{obj:25s}: min={df[obj].min():.4f}, max={df[obj].max():.4f}, "
                  f"mean={df[obj].mean():.4f}, std={df[obj].std():.4f}")

    print("\nHyperparameter Statistics:")
    print("-" * 80)

    hparams = ["learning_rate", "weight_decay", "dropout_rate",
               "augmentation_strength", "unfreeze_fraction"]
    for hp in hparams:
        if hp in df.columns:
            if hp in ["learning_rate", "weight_decay"]:
                # Log scale
                print(f"{hp:25s}: min={df[hp].min():.2e}, max={df[hp].max():.2e}, "
                      f"mean={df[hp].mean():.2e}")
            else:
                print(f"{hp:25s}: min={df[hp].min():.4f}, max={df[hp].max():.4f}, "
                      f"mean={df[hp].mean():.4f}")


def find_extreme_solutions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find extreme solutions (best for each objective).

    Args:
        df: Pareto solutions DataFrame

    Returns:
        DataFrame with extreme solutions
    """
    extreme_solutions = []

    # Best PR-AUC
    best_pr_auc_idx = df["pr_auc"].idxmax()
    extreme_solutions.append({
        "criterion": "Best PR-AUC",
        "solution_id": df.loc[best_pr_auc_idx, "solution_id"],
        "pr_auc": df.loc[best_pr_auc_idx, "pr_auc"],
        "auroc": df.loc[best_pr_auc_idx, "auroc"],
        "brier": df.loc[best_pr_auc_idx, "brier"],
        "robustness": df.loc[best_pr_auc_idx, "robustness_degradation"],
    })

    # Best AUROC
    best_auroc_idx = df["auroc"].idxmax()
    extreme_solutions.append({
        "criterion": "Best AUROC",
        "solution_id": df.loc[best_auroc_idx, "solution_id"],
        "pr_auc": df.loc[best_auroc_idx, "pr_auc"],
        "auroc": df.loc[best_auroc_idx, "auroc"],
        "brier": df.loc[best_auroc_idx, "brier"],
        "robustness": df.loc[best_auroc_idx, "robustness_degradation"],
    })

    # Best Brier (minimum)
    best_brier_idx = df["brier"].idxmin()
    extreme_solutions.append({
        "criterion": "Best Brier",
        "solution_id": df.loc[best_brier_idx, "solution_id"],
        "pr_auc": df.loc[best_brier_idx, "pr_auc"],
        "auroc": df.loc[best_brier_idx, "auroc"],
        "brier": df.loc[best_brier_idx, "brier"],
        "robustness": df.loc[best_brier_idx, "robustness_degradation"],
    })

    # Best Robustness (minimum degradation)
    best_robust_idx = df["robustness_degradation"].idxmin()
    extreme_solutions.append({
        "criterion": "Best Robustness",
        "solution_id": df.loc[best_robust_idx, "solution_id"],
        "pr_auc": df.loc[best_robust_idx, "pr_auc"],
        "auroc": df.loc[best_robust_idx, "auroc"],
        "brier": df.loc[best_robust_idx, "brier"],
        "robustness": df.loc[best_robust_idx, "robustness_degradation"],
    })

    return pd.DataFrame(extreme_solutions)


def plot_pareto_front_2d(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create 2D scatter plots of objective pairs.

    Args:
        df: Pareto solutions DataFrame
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    objective_pairs = [
        ("pr_auc", "auroc"),
        ("pr_auc", "brier"),
        ("pr_auc", "robustness_degradation"),
        ("auroc", "brier"),
        ("auroc", "robustness_degradation"),
        ("brier", "robustness_degradation"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (obj1, obj2) in enumerate(objective_pairs):
        ax = axes[idx]
        ax.scatter(df[obj1], df[obj2], alpha=0.6, s=50)
        ax.set_xlabel(obj1.replace("_", " ").title())
        ax.set_ylabel(obj2.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_front_2d.png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved 2D Pareto front plots to: {output_dir}/pareto_front_2d.png")
    plt.close()


def plot_hyperparameter_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot distributions of hyperparameters in Pareto front.

    Args:
        df: Pareto solutions DataFrame
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    hparams = ["learning_rate", "weight_decay", "dropout_rate",
               "augmentation_strength", "unfreeze_fraction"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, hp in enumerate(hparams):
        ax = axes[idx]
        if hp in ["learning_rate", "weight_decay"]:
            # Log scale
            ax.hist(np.log10(df[hp]), bins=20, alpha=0.7, edgecolor="black")
            ax.set_xlabel(f"log10({hp})")
        else:
            ax.hist(df[hp], bins=20, alpha=0.7, edgecolor="black")
            ax.set_xlabel(hp.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hyperparameter_distributions.png"),
                dpi=300, bbox_inches="tight")
    print(f"Saved hyperparameter distributions to: {output_dir}/hyperparameter_distributions.png")
    plt.close()


def export_solution_configs(df: pd.DataFrame, solution_ids: List[int], output_dir: str) -> None:
    """
    Export hyperparameter configurations for specific solutions.

    Args:
        df: Pareto solutions DataFrame
        solution_ids: List of solution IDs to export
        output_dir: Directory to save configuration files
    """
    os.makedirs(output_dir, exist_ok=True)

    hparam_cols = ["learning_rate", "weight_decay", "dropout_rate",
                   "augmentation_strength", "unfreeze_fraction"]

    for sol_id in solution_ids:
        solution = df[df["solution_id"] == sol_id]
        if len(solution) == 0:
            print(f"Warning: Solution ID {sol_id} not found")
            continue

        config = {hp: float(solution[hp].iloc[0]) for hp in hparam_cols}

        output_path = os.path.join(output_dir, f"solution_{sol_id}_config.json")
        import json
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Exported solution {sol_id} config to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Pareto front from NSGA-III optimization")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./optimization_results",
        help="Directory containing optimization results",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp of results file (optional, uses most recent if not specified)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pareto_analysis",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--export_solutions",
        type=int,
        nargs="+",
        default=None,
        help="Solution IDs to export configurations for",
    )
    args = parser.parse_args()

    # Load Pareto solutions
    df = load_pareto_solutions(args.results_dir, args.timestamp)

    # Print summary statistics
    print_summary_statistics(df)

    # Find extreme solutions
    print("\n" + "=" * 80)
    print("EXTREME SOLUTIONS")
    print("=" * 80)
    extreme_df = find_extreme_solutions(df)
    print(extreme_df.to_string(index=False))

    # Create visualizations
    plot_pareto_front_2d(df, args.output_dir)
    plot_hyperparameter_distributions(df, args.output_dir)

    # Export solution configurations
    if args.export_solutions is not None:
        export_solution_configs(df, args.export_solutions, args.output_dir)
    else:
        # Export extreme solutions by default
        extreme_ids = extreme_df["solution_id"].astype(int).tolist()
        export_solution_configs(df, extreme_ids, args.output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
