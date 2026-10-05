# TabLint demos

Start with `uv run --extra notebook tablint demo` from the repository root for interactive Jupyter review. Choose **Run → Run All Cells** in the notebook.

The saved demo needs no model weights or GPU. VS Code (`npm run demo:vscode`) and the browser viewer (`uv run --extra demo tablint app`) are optional integrations.

- [Jupyter demo notebook](../examples/tablint_demo.ipynb) — inspect cells and retrieve cleaned DataFrames and decisions.
- [Auto MPG data and saved reports](video/README.md) — no inference needed to replay.
- [Approved product video](showcase/README.md) — 1:28 animation and provenance.
- [Getting started](../docs/GETTING_STARTED.md) — setup and live inference.

The separate `app.py` is the original Chain of Custody research viewer; see [its documentation](RESEARCH_VIEWER.md). The TabLint browser command opens `proofread_app.py`.
