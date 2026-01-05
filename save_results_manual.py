"""
Manual result saving cell for Colab.
Use this if optimization completed but saving failed.
"""

cell_code = """
# ============================================================================
# MANUALLY SAVE OPTIMIZATION RESULTS (Use if auto-save failed)
# ============================================================================

import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Create timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 80)
print("SAVING OPTIMIZATION RESULTS")
print("=" * 80)

# Extract Pareto solutions and objectives from result
pareto_solutions = result.X.copy()  # Hyperparameter values
pareto_objectives = result.F.copy()  # Objective values

print(f"Pareto front size: {len(pareto_solutions)} solutions")
print()

# ============================================================================
# 1. SAVE PARETO FRONT AS CSV (Human-readable)
# ============================================================================

# Transform hyperparameters back to original scale
rows = []
for i, (x, f) in enumerate(zip(pareto_solutions, pareto_objectives)):
    # Decode hyperparameters (log scale for lr and wd)
    learning_rate = 10 ** x[0]
    weight_decay = 10 ** x[1]
    dropout_rate = x[2]
    augmentation_strength = x[3]
    unfreeze_fraction = x[4]

    # Extract objectives (convert minimization back to original metrics)
    pr_auc = -f[0]  # Was negated for minimization
    auroc = -f[1]   # Was negated for minimization
    brier = f[2]
    robustness_degradation = f[3]

    row = {
        "solution_id": i,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "augmentation_strength": augmentation_strength,
        "unfreeze_fraction": unfreeze_fraction,
        "pr_auc": pr_auc,
        "auroc": auroc,
        "brier": brier,
        "robustness_degradation": robustness_degradation,
    }
    rows.append(row)

pareto_df = pd.DataFrame(rows)

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, f"pareto_solutions_{timestamp}.csv")
pareto_df.to_csv(csv_path, index=False)
print(f"[OK] Saved Pareto front CSV: {csv_path}")
print()

# ============================================================================
# 2. SAVE METADATA AS JSON
# ============================================================================

metadata = {
    "timestamp": timestamp,
    "duration_seconds": (datetime.now() - runner.start_time).total_seconds() if hasattr(runner, 'start_time') else None,
    "n_solutions": len(pareto_solutions),
    "pop_size": 24,
    "n_generations": 50,
    "n_objectives": 4,
    "random_seed": 42,
    "max_epochs": 3,  # Update if different
}

metadata_path = os.path.join(OUTPUT_DIR, f"metadata_{timestamp}.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[OK] Saved metadata JSON: {metadata_path}")
print()

# ============================================================================
# 3. SAVE RESULT DATA AS PICKLE (Numpy arrays only - picklable!)
# ============================================================================

result_data = {
    "X": pareto_solutions,  # Pareto solutions (hyperparameters)
    "F": pareto_objectives,  # Pareto objectives
    "timestamp": timestamp,
    "duration_seconds": metadata["duration_seconds"],
}

result_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.pkl")
with open(result_path, "wb") as f:
    pickle.dump(result_data, f)
print(f"[OK] Saved result pickle: {result_path}")
print()

# ============================================================================
# 4. DISPLAY SUMMARY
# ============================================================================

print("=" * 80)
print("RESULTS SAVED SUCCESSFULLY!")
print("=" * 80)
print(f"Location: {OUTPUT_DIR}")
print()
print("Files created:")
print(f"  1. {os.path.basename(csv_path)} - Pareto front (human-readable)")
print(f"  2. {os.path.basename(metadata_path)} - Optimization metadata")
print(f"  3. {os.path.basename(result_path)} - Result arrays (pickle)")
print()

# Show summary statistics
print("Pareto Front Summary:")
print(f"  Solutions: {len(pareto_df)}")
print(f"  Best PR-AUC: {pareto_df['pr_auc'].max():.4f}")
print(f"  Best AUROC: {pareto_df['auroc'].max():.4f}")
print(f"  Best Brier: {pareto_df['brier'].min():.4f}")
print(f"  Best Robustness: {pareto_df['robustness_degradation'].min():.4f}")
print()

print("You can now:")
print("  - Download the CSV file to analyze solutions")
print("  - Use pareto_df to explore results in this notebook")
print("  - Access files anytime from Google Drive")
print("=" * 80)

# Make pareto_df available for further analysis
print()
print("Variable 'pareto_df' is now available for analysis.")
print("Example: pareto_df.head()")
"""

print("=" * 80)
print("COPY THIS CELL TO COLAB")
print("=" * 80)
print()
print(cell_code)
print()
print("=" * 80)
print("INSTRUCTIONS:")
print("=" * 80)
print("1. Create a new code cell in Colab")
print("2. Copy and paste the code above")
print("3. Run the cell")
print()
print("This will save:")
print("  - pareto_solutions_TIMESTAMP.csv (human-readable)")
print("  - metadata_TIMESTAMP.json (optimization info)")
print("  - result_TIMESTAMP.pkl (picklable arrays)")
print()
print("All files saved to Google Drive for permanent storage!")
