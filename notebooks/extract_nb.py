import json

with open('notebooks/code.ipynb', 'r') as f:
    nb = json.load(f)

with open('notebooks/code_extracted.py', 'w') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            f.write("".join(cell['source']) + "\n\n")
