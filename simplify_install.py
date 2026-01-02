import json

# Read notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the install dependencies cell
for i, cell in enumerate(nb['cells']):
    if cell.get('metadata', {}).get('id') == 'install_deps':
        # Simplify - just install requirements
        cell['source'] = [
            "# Install required packages\n",
            "!pip install -q -r requirements.txt\n",
            "\n",
            'print("\\n[SUCCESS] All dependencies installed!")'
        ]
        print(f"Updated cell {i} (install_deps) - removed pip install -e .")
        break

# Write updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook simplified - using sys.path instead of package installation")
