import json

# Read notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the verify setup cell
for i, cell in enumerate(nb['cells']):
    if cell.get('metadata', {}).get('id') == 'verify_setup':
        # Update the cell content to add path setup
        cell['source'] = [
            "import sys\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "# Add project root to Python path\n",
            "project_root = os.getcwd()\n",
            "if project_root not in sys.path:\n",
            "    sys.path.insert(0, project_root)\n",
            "\n",
            "print(f\"Project root: {project_root}\")\n",
            "print(f\"Python path updated: {project_root in sys.path}\")\n",
            "\n",
            "import config\n",
            "from optimization.nsga3_runner import load_metadata\n",
            "\n",
            "# Load VinDr-Mammo metadata\n",
            "print(\"\\nLoading VinDr-Mammo metadata...\")\n",
            "vindr_metadata = load_metadata(\n",
            "    dataset_name=\"vindr\",\n",
            "    dataset_path=config.VINDR_MAMMO_PATH,\n",
            "    dataset_config=config.VINDR_CONFIG\n",
            ")\n",
            "print(f\"[OK] Loaded {len(vindr_metadata)} images\")\n",
            "print(f\"     Patients: {vindr_metadata['patient_id'].nunique()}\")\n",
            "print(f\"     Label distribution: {vindr_metadata['label'].value_counts().to_dict()}\")\n",
            "\n",
            "# Load INbreast metadata\n",
            "print(\"\\nLoading INbreast metadata...\")\n",
            "inbreast_metadata = load_metadata(\n",
            "    dataset_name=\"inbreast\",\n",
            "    dataset_path=config.INBREAST_PATH,\n",
            "    dataset_config=config.INBREAST_CONFIG\n",
            ")\n",
            "print(f\"[OK] Loaded {len(inbreast_metadata)} images\")\n",
            "print(f\"     Patients: {inbreast_metadata['patient_id'].nunique()}\")\n",
            "print(f\"     Label distribution: {inbreast_metadata['label'].value_counts().to_dict()}\")\n",
            "\n",
            "print(\"\\n[SUCCESS] Setup verification complete!\")"
        ]
        print(f"Updated cell {i} (verify_setup)")
        break

# Also update the optimization cell to ensure path is set
for i, cell in enumerate(nb['cells']):
    if cell.get('metadata', {}).get('id') == 'run_optimization':
        # Ensure the cell starts with path setup
        current_source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'sys.path.insert' not in current_source:
            cell['source'] = [
                "import sys\n",
                "import os\n",
                "\n",
                "# Ensure project root is in path\n",
                "project_root = os.getcwd()\n",
                "if project_root not in sys.path:\n",
                "    sys.path.insert(0, project_root)\n",
                "\n",
                "from optimization.nsga3_runner import NSGA3Runner\n",
                "from data.dataset import create_train_val_split\n",
                "from pathlib import Path\n",
                "import config\n",
                "\n",
                "# Create train/val split\n",
                "print(\"Creating train/validation split...\")\n",
                "train_metadata, val_metadata = create_train_val_split(vindr_metadata)\n",
                "\n",
                "print(f\"Train samples: {len(train_metadata)}\")\n",
                "print(f\"Validation samples: {len(val_metadata)}\")\n",
                "print(f\"Unique patients - Train: {train_metadata['patient_id'].nunique()}, \"\n",
                "      f\"Val: {val_metadata['patient_id'].nunique()}\")\n",
                "\n",
                "# Create runner with checkpoint saving\n",
                "print(\"\\nInitializing NSGA-III runner...\")\n",
                "image_dir = str(Path(config.VINDR_MAMMO_PATH) / config.VINDR_CONFIG[\"image_dir\"])\n",
                "runner = NSGA3Runner(\n",
                "    train_metadata=train_metadata,\n",
                "    val_metadata=val_metadata,\n",
                "    image_dir=image_dir,\n",
                "    output_dir=\"./optimization_results\",\n",
                "    checkpoint_dir=\"./checkpoints\",\n",
                "    save_frequency=1  # Save every generation\n",
                ")\n",
                "\n",
                "print(\"\\n\" + \"=\"*80)\n",
                "print(\"STARTING OPTIMIZATION\")\n",
                "print(\"=\"*80)\n",
                "print(\"This may take a while depending on:\")\n",
                "print(\"  - Population size (current: from config)\")\n",
                "print(\"  - Number of generations (current: from config)\")\n",
                "print(\"  - Dataset size\")\n",
                "print(\"  - GPU availability\")\n",
                "print(\"\\nCheckpoints will be saved every generation.\")\n",
                "print(\"You can monitor progress in the optimization_checkpoints folder.\")\n",
                "print(\"=\"*80 + \"\\n\")\n",
                "\n",
                "# Run optimization\n",
                "result = runner.run()\n",
                "\n",
                "print(\"\\n\" + \"=\"*80)\n",
                "print(\"OPTIMIZATION COMPLETE!\")\n",
                "print(\"=\"*80)\n",
                "print(f\"Pareto front size: {len(result.F)}\")\n",
                "print(f\"Results saved to: {runner.output_dir}\")\n",
                "print(\"=\"*80)"
            ]
            print(f"Updated cell {i} (run_optimization)")
            break

# Write updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nNotebook updated successfully!")
print("All cells now include proper path setup for imports.")
