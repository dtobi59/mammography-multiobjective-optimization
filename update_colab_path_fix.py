import json

# Read notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the clone repo cell and add path setup right after
for i, cell in enumerate(nb['cells']):
    if cell.get('metadata', {}).get('id') == 'clone_repo':
        # Add a new cell right after this one to setup the path
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "id": "setup_python_path"
            },
            "outputs": [],
            "source": [
                "# Setup Python path to ensure imports work\n",
                "import sys\n",
                "import os\n",
                "\n",
                "# Get current directory (project root)\n",
                "project_root = os.getcwd()\n",
                "print(f\"Project root: {project_root}\")\n",
                "\n",
                "# Add to Python path if not already there\n",
                "if project_root not in sys.path:\n",
                "    sys.path.insert(0, project_root)\n",
                "    print(f\"Added {project_root} to sys.path\")\n",
                "\n",
                "# Verify path setup\n",
                "print(f\"\\nPython sys.path[0]: {sys.path[0]}\")\n",
                "print(\"[OK] Path setup complete!\")"
            ]
        }
        # Insert after current cell
        nb['cells'].insert(i+1, new_cell)
        print(f"Added new path setup cell after cell {i}")
        break

# Write updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with explicit path setup!")
