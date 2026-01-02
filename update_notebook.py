import json

# Read notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the install dependencies cell (cell index 6)
for i, cell in enumerate(nb['cells']):
    if cell.get('metadata', {}).get('id') == 'install_deps':
        # Update the cell content
        cell['source'] = [
            "# Install required packages\n",
            "!pip install -q -r requirements.txt\n",
            "\n",
            "# Install the package in editable mode to fix imports\n",
            "!pip install -q -e .\n",
            "\n",
            'print("\\n[SUCCESS] All dependencies installed!")'
        ]
        print(f"Updated cell {i} (install_deps)")
        break

# Write updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully!")
