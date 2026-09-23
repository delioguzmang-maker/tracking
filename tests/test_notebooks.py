"""Every notebook code cell must at least compile (catches broken cells before a user opens them)."""
import ast
import json
from pathlib import Path

import pytest

NB = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NB, ids=[p.name for p in NB])
def test_notebook_cells_compile(path):
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # IPython magics / shell escapes are not Python
        src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))
        ast.parse(src, filename=f"{path.name}[cell {i}]")
