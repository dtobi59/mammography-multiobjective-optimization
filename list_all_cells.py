import json

with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for i, cell in enumerate(notebook['cells']):
    source = ''.join(cell.get('source', []))
    cell_type = cell.get('cell_type', 'unknown')
    cell_id = cell.get('id', 'unknown')

    # Show first 100 chars
    preview = source[:100].replace('\n', ' ')
    print(f"Cell {i} ({cell_type}, {cell_id}): {preview}")
