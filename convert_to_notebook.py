"""Convert the Python script to a Jupyter Notebook (.ipynb)."""
import json
import re

def py_to_notebook(py_file, nb_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []

    # Split by "# %%" markers
    raw_cells = re.split(r'^# %%', content, flags=re.MULTILINE)

    for i, cell_content in enumerate(raw_cells):
        cell_content = cell_content.strip()
        if not cell_content:
            continue

        # Check if it's a markdown cell
        if cell_content.startswith(' [markdown]'):
            # Extract markdown content from comment lines
            lines = cell_content.split('\n')[1:]  # skip the [markdown] line
            md_lines = []
            for line in lines:
                if line.startswith('# '):
                    md_lines.append(line[2:])
                elif line.startswith('#'):
                    md_lines.append(line[1:])
                else:
                    break
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [l + '\n' for l in md_lines]
            })
        else:
            # Code cell
            # Remove leading newline if present
            if cell_content.startswith('\n'):
                cell_content = cell_content[1:]

            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + '\n' for l in cell_content.split('\n')]
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
                "mimetype": "text/x-python",
                "file_extension": ".py"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Notebook saved to: {nb_file}")
    print(f"Total cells: {len(cells)}")

if __name__ == '__main__':
    py_to_notebook(
        r'c:\Users\Ayush Singh\Desktop\final minor project\Adaptive_Graph_Transformer_Energy_Forecasting.py',
        r'c:\Users\Ayush Singh\Desktop\final minor project\Adaptive_Graph_Transformer_Energy_Forecasting.ipynb'
    )
