"""
Add testing configuration section to Colab notebook.
Allows easy override of MAX_EPOCHS and other settings.
"""

import json

# Read the notebook
with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find Section 5a markdown header
section_5a_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if '### 5a. Start New Optimization' in source:
            section_5a_idx = i
            break

if section_5a_idx is None:
    print("[ERROR] Could not find Section 5a header")
    exit(1)

print(f"Found Section 5a header at index {section_5a_idx}")

# Create new markdown cell for testing configuration
testing_config_markdown = {
    "cell_type": "markdown",
    "id": "cell-5a-testing-config-header",
    "metadata": {},
    "source": [
        "#### Testing Configuration (Optional)\n\n",
        "**For quick testing**, you can reduce the number of epochs here.\n\n",
        "**Skip this cell for production runs** (uses default: 100 epochs)."
    ]
}

# Create new code cell for testing configuration
testing_config_code = {
    "cell_type": "code",
    "id": "cell-5a-testing-config",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# OPTIONAL: OVERRIDE SETTINGS FOR TESTING\n",
        "# ============================================================================\n",
        "# Uncomment and modify these lines to speed up testing:\n",
        "\n",
        "import config\n",
        "\n",
        "# Set epochs to 3 for quick testing (default: 100)\n",
        "config.MAX_EPOCHS = 3\n",
        "print(f\"[TESTING MODE] MAX_EPOCHS set to {config.MAX_EPOCHS}\")\n",
        "\n",
        "# Optional: Reduce early stopping patience (default: 15)\n",
        "# config.EARLY_STOPPING_PATIENCE = 3\n",
        "# print(f\"[TESTING MODE] EARLY_STOPPING_PATIENCE set to {config.EARLY_STOPPING_PATIENCE}\")\n",
        "\n",
        "print(\"\\n[WARNING] Testing mode enabled - models will train for only 3 epochs!\")\n",
        "print(\"For production, comment out this cell or set MAX_EPOCHS = 100\")\n"
    ]
}

# Insert the two new cells right after Section 5a header
insert_idx = section_5a_idx + 1
notebook['cells'].insert(insert_idx, testing_config_markdown)
notebook['cells'].insert(insert_idx + 1, testing_config_code)

# Save updated notebook
with open('colab_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print()
print("[OK] Added testing configuration section to notebook!")
print()
print("Added 2 new cells:")
print(f"  1. Index {insert_idx}: Markdown header for testing config")
print(f"  2. Index {insert_idx + 1}: Code cell with MAX_EPOCHS override")
print()
print("Usage in Colab:")
print("  - Run this cell to set MAX_EPOCHS = 3 for testing")
print("  - Skip this cell for production (uses default MAX_EPOCHS = 100)")
print("  - Users can easily modify the value as needed")
