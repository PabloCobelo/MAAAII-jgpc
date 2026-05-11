import json
import sys

with open('P2_MAA2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 80)
print("NOTEBOOK STRUCTURE AND CONTENT")
print("=" * 80)

for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source = ''.join(cell['source'])
    
    if cell_type == 'markdown':
        print(f"\n[CELL {i}] MARKDOWN SECTION")
        print("-" * 80)
        print(source[:500] if len(source) > 500 else source)
        if len(source) > 500:
            print(f"... (truncated, total {len(source)} chars)")
    else:
        print(f"\n[CELL {i}] CODE")
        print("-" * 80)
        lines = source.split('\n')
        print(f"Lines: {len(lines)}")
        print("First 20 lines:")
        for line in lines[:20]:
            print(line)
        if len(lines) > 20:
            print(f"... ({len(lines) - 20} more lines)")
