"""
Add prominent warning before Section 5a about running Section 4a first.
"""

import json

# Read the notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find Section 5 header (markdown before Section 5a)
section5_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if '## 5. Run NSGA-III Optimization' in source:
            section5_index = i
            break

if section5_index is None:
    print("[ERROR] Could not find Section 5 header")
    exit(1)

print(f"Found Section 5 header at index {section5_index}")

# Update Section 5 header to include prominent warning
updated_section5_header = {
    "cell_type": "markdown",
    "id": notebook['cells'][section5_index].get('id', None),
    "metadata": {},
    "source": [
        "## 5. Run NSGA-III Optimization\n\n",
        "This will optimize 5 hyperparameters for 4 objectives with automatic checkpointing.\n\n",
        "**IMPORTANT: You MUST run Section 4a first!**\n\n",
        "Section 4a creates the stratified subsample (train_metadata, val_metadata) required for optimization.\n\n",
        "**If your session timed out, skip to Section 5b to resume from checkpoint.**"
    ]
}

# Replace the old header
notebook['cells'][section5_index] = updated_section5_header

# Also update Section 5a header
section5a_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if '### 5a. Start New Optimization' in source or '5a' in source:
            section5a_index = i
            break

if section5a_index:
    print(f"Found Section 5a header at index {section5a_index}")

    updated_section5a_header = {
        "cell_type": "markdown",
        "id": notebook['cells'][section5a_index].get('id', None),
        "metadata": {},
        "source": [
            "### 5a. Start New Optimization\n\n",
            "**PREREQUISITE: Run Section 4a first!**\n\n",
            "This section requires `train_metadata` and `val_metadata` from Section 4a.\n\n",
            "**Step 1:** Run the cell below to verify your data and setup A100 optimizations.\n\n",
            "**Step 2:** If verification passes, run the next cell to start optimization."
        ]
    }

    notebook['cells'][section5a_index] = updated_section5a_header

# Save updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print()
print("[OK] Added warnings to Section 5 headers!")
print()
print("Updated headers:")
print("  - Section 5: Added 'IMPORTANT: You MUST run Section 4a first!'")
print("  - Section 5a: Added 'PREREQUISITE: Run Section 4a first!'")
print()
print("Users will now clearly see they need to run Section 4a before Section 5a.")
