from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / 'outputs' / 'student_support' / 'student_support_queue.ipynb'
nb = nbformat.read(path, as_version=4)
start = time.perf_counter()
def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executing cell {cell_index + 1}/{len(nb.cells)}', flush=True)

client = NotebookClient(nb, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}},
                        on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print(f'Executed notebook in {time.perf_counter() - start:.1f} seconds', flush=True)
