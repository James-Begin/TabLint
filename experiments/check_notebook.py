"""Execute examples/proofread_notebook.ipynb in a real Jupyter kernel; fail on any error; report what rendered."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[1]
import sys
name = sys.argv[1] if len(sys.argv) > 1 else "proofread_notebook.ipynb"
nb = nbformat.read(ROOT / "examples" / name, as_version=4)
NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "examples")}}).execute()
for c in nb.cells:
    if c.cell_type != "code":
        continue
    kinds = [list(o.get("data", {}).keys()) for o in c.outputs if o.output_type in ("execute_result", "display_data")]
    errs = [o for o in c.outputs if o.output_type == "error"]
    assert not errs, errs
    text = [o.get("text", "") for o in c.outputs if o.output_type == "stream"]
    print(c.source.splitlines()[0][:60], "->", kinds, ("| " + " ".join(text).strip()[:300]) if text else "")
print("notebook executed without errors")
