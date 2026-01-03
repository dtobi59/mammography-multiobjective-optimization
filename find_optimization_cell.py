import json

with open('colab_tutorial.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for i, cell in enumerate(notebook['cells']):
    source = ''.join(cell.get('source', []))
    if 'runner = NSGA3Runner' in source or 'runner.run()' in source:
        print(f"\n=== Cell {i} ===")
        print(source[:500])
        print(f"\n... (cell ID: {cell.get('id', 'unknown')})")
