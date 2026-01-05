"""
Test each cell in zero_shot_evaluation.ipynb for syntax and logic errors.
"""

import json
import ast

# Read notebook
with open('zero_shot_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 80)
print("TESTING NOTEBOOK CELLS FOR ERRORS")
print("=" * 80)
print()

errors_found = []
warnings_found = []

for i, cell in enumerate(nb['cells']):
    cell_id = cell.get('id', f'cell_{i}')
    cell_type = cell.get('cell_type', 'unknown')

    if cell_type != 'code':
        continue

    source = ''.join(cell.get('source', []))

    # Skip shell commands and magic commands (check all lines)
    lines = [line.strip() for line in source.split('\n') if line.strip()]
    if any(line.startswith('!') or line.startswith('%') for line in lines):
        print(f"[SKIP] {cell_id} - Shell/magic command")
        continue

    # Skip empty cells
    if not source.strip():
        print(f"[SKIP] {cell_id} - Empty cell")
        continue

    print(f"[TEST] {cell_id}")

    # Test for syntax errors
    try:
        ast.parse(source)
        print(f"  [OK] Syntax valid")
    except SyntaxError as e:
        errors_found.append({
            'cell_id': cell_id,
            'error_type': 'SyntaxError',
            'message': str(e),
            'line': e.lineno
        })
        print(f"  [ERROR] SyntaxError at line {e.lineno}: {e.msg}")
        continue

    # Check for common issues
    issues = []

    # Check for undefined variables (basic check)
    if 'pareto_df' in source and cell_id != 'cell-list-results' and 'cell-list' not in cell_id:
        if i < 10:  # Early in notebook
            issues.append("Uses 'pareto_df' before it's defined")

    if 'inbreast_metadata' in source and 'cell-load-inbreast' not in cell_id:
        if i < 14:
            issues.append("Uses 'inbreast_metadata' before it's defined")

    if 'selected_solution' in source and 'cell-select' not in cell_id:
        if i < 17:
            issues.append("Uses 'selected_solution' before it's defined")

    # Check for img_path usage
    if 'img_path.exists()' in source and 'img_path =' not in source:
        issues.append("Uses 'img_path' without initialization")

    # Check imports
    if 'from google.colab' in source:
        print(f"  [INFO] Colab-specific import (expected)")

    if issues:
        for issue in issues:
            warnings_found.append({
                'cell_id': cell_id,
                'warning': issue
            })
            print(f"  [WARNING] {issue}")

    if not issues:
        print(f"  [OK] No logical issues detected")

    print()

# Summary
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Errors found: {len(errors_found)}")
print(f"Warnings found: {len(warnings_found)}")
print()

if errors_found:
    print("ERRORS:")
    for err in errors_found:
        print(f"  Cell: {err['cell_id']}")
        print(f"  Type: {err['error_type']}")
        print(f"  Line: {err.get('line', 'N/A')}")
        print(f"  Message: {err['message']}")
        print()

if warnings_found:
    print("WARNINGS:")
    for warn in warnings_found:
        print(f"  Cell: {warn['cell_id']}")
        print(f"  Issue: {warn['warning']}")
    print()

if errors_found:
    print("[FAILED] Notebook has errors that need to be fixed!")
    exit(1)
else:
    print("[PASSED] All code cells have valid Python syntax!")
