# Getting started with TabLint

## Jupyter notebooks

### Install in your existing notebook

Use a **Python 3.12** kernel and have Git installed. In your notebook, install from GitHub:

```python
%pip install "tabpfn-proofread[notebook] @ git+https://github.com/James-Begin/TabLint.git"
```

Restart the kernel after installation. `%pip` installs into the active notebook kernel's environment. The project includes the Python library, interactive widget and JupyterLab.

From a terminal using the same Python environment, the equivalent is:

```sh
python -m pip install "tabpfn-proofread[notebook] @ git+https://github.com/James-Begin/TabLint.git"
```

### Create a new notebook environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/James-Begin/TabLint.git && cd TabLint && uv run --extra notebook jupyter lab
```

`uv` prepares Python 3.12 and the locked dependencies. JupyterLab opens in your browser. Create a **Python 3 (ipykernel)** notebook. This opens a notebook environment for your own work; it does not run analysis automatically.

For headless use: `uv run --extra notebook jupyter lab --no-browser --port 8888`. Follow the local URL printed by Jupyter and stop the server with Ctrl+C. [JupyterLab startup](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html) · [uv's notebook guide](https://docs.astral.sh/uv/guides/integration/jupyter/)

### Check your DataFrame

```python
import pandas as pd
import proofread

df = pd.read_csv("data.csv")  # use the path to your own CSV
review = df.tablint.view(categorical=True)
review
```

Click highlighted cells for the expected value, uncertainty, likely cause and related columns. **Accept** applies a suggestion to a cleaned copy, **dismiss** leaves the source value unchanged, and **undo** reverses the last widget decision. Keyboard shortcuts after clicking the widget: `n` / `p` next / previous, `a` accept, `d` dismiss, `u` undo.

Numeric cells use TabPFN's predictive distribution. Low-cardinality categorical cells use its classifier probabilities and show likely alternatives. High-cardinality identifiers and free text are excluded from categorical checks. Add `label="outcome"` to check a known target column.

To inspect the report before opening the widget:

```python
report = df.tablint.check(categorical=True)
report.issues
report.highlight()
review = report.widget()
review
```

### Export your results

After making decisions, run another cell:

```python
cleaned = review.cleaned
changes = review.decisions
cleaned.to_csv("cleaned.csv", index=False)
changes.to_csv("decisions.csv", index=False)
```

Rerun this cell after further review to refresh the exported results. The original DataFrame stays intact. Decisions contain recorded and suggested values, model reasons and the current review state. They are a snapshot; undo history lasts for the widget session. The optional VS Code integration separately has a persistent ledger with timestamps and flag-for-review actions.

The original `df.proofread` accessor remains supported. `%load_ext proofread` registers both `%tablint df` and the original `%proofread df` magic.

## Model setup

First inference downloads TabPFN weights and may open the upstream model-license acceptance flow. Complete that flow if you agree to the model terms. Weights are cached separately and are not redistributed with TabLint. A GPU is optional; CPU inference can take several minutes.

Pass `fast=True` to use TabPFN-3.5-Fast, `threshold=3` to raise the numerical flag threshold, or `categorical=False` to skip categorical-column checks. A flag is a prompt to verify the source record, not a confirmed error.

## VS Code

Jupyter is the primary workflow; the editor integration is optional.

From a cloned checkout, install Node.js 22+, npm and VS Code 1.90+. Build the extension:

```sh
npm --prefix vscode-proofread ci
mkdir -p dist
npm --prefix vscode-proofread run package
```

In VS Code run **Extensions: Install from VSIX…** and choose `dist/tablint.vsix`. Alternatively, install the [built extension](../demo/showcase/tablint.vsix). Open your CSV and run **TabLint: Check this CSV with TabPFN-3.5**.

Open this repository as the workspace, or set `proofread.command` to an absolute installed CLI command or `uv run --project /path/to/TabLint tablint`. Hover flagged cells and use Quick Fix (`Ctrl+.` / `Cmd+.`) to replace, flag or dismiss a suggestion. **TabLint: Open decision ledger** lets you inspect reasons and undo decisions. [Extension settings](../vscode-proofread/README.md#settings)

## Terminal

From the cloned checkout:

```sh
uv run tablint check data.csv --out reports/data
uv run tablint view reports/data.json
```

The check writes JSON, Markdown, highlighted HTML and an issues CSV. The terminal viewer supports `n` / `p` next / previous, `a` accept, `d` dismiss, `u` undo, `f` flagged rows, and `s` save a cleaned CSV and decisions JSON.

## Browser viewer

```sh
uv run --extra demo tablint app
```

Upload your CSV and press **Check with TabLint**. This performs a live check and needs the model weights. The command opens `demo/proofread_app.py`; `demo/app.py` is a separate research audit viewer.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Python version mismatch | Select a Python 3.12 kernel, or use the `uv` setup command to prepare that interpreter. |
| Git is missing during installation | Install Git, or clone/download the repository and run `python -m pip install -e ".[notebook]"` from its root. |
| Widget is blank or `anywidget` cannot be imported | Install in the active kernel's environment, restart the kernel, and rerun the widget cell. |
| `proofread` cannot be imported | Confirm the selected kernel is the environment where you installed TabLint. |
| Port 8888 is busy | Jupyter normally chooses the next free port; follow the printed URL or pass `--port 8890`. |
| Model access/download error | Complete the upstream model-license flow and check network access. |
| Slow analysis | Use a CUDA GPU or try `fast=True`. More columns and folds increase inference work. |
| VS Code shows no diagnostics | Check the CLI path, run the CSV analysis command, and inspect its output. Modified cells intentionally stop showing stale diagnostics. |

[Back to the README](../README.md)
