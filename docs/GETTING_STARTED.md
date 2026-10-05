# Getting started with TabLint

## Jupyter notebooks

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/James-Begin/TabLint.git && cd TabLint && uv run --extra notebook tablint demo
```

`uv` prepares Python 3.12 and the locked notebook dependencies. The launcher starts JupyterLab with the same Python environment, copies the [demo notebook](../examples/tablint_demo.ipynb) to `demo/notebook-workspace/TabLint-demo.ipynb` on first use, and opens that copy. It preserves edits on subsequent launches. It serves locally on `127.0.0.1` with Jupyter's normal token authentication.

Choose **Run → Run All Cells**. The demo replays saved Auto MPG predictions, so it does not download model weights or run inference. Click highlighted cells to see expected values, uncertainty, likely causes and row context; accept, dismiss or undo suggestions. The widget supports numerical, categorical and label issues; this stored report contains numerical and origin-label issues.

```python
cleaned = review.cleaned       # accepted suggestions applied to a copy
changes = review.decisions     # current decisions and model reasons
```

Rerun the results cell after making decisions. The original DataFrame and report stay intact. Export `cleaned` and `changes` to CSV to retain your work; the notebook shows the code. Decisions are a current-state snapshot, and undo history lasts for the widget session. The VS Code integration separately has a persistent ledger with timestamps and flag-for-review actions.

For remote/headless use: `uv run --extra notebook tablint demo --no-browser --port 8888`. Use the local URL printed by Jupyter; stop the server with Ctrl+C. `--prepare-only` copies the notebook without starting a server. To start a fresh notebook without replacing your work, open `examples/tablint_demo.ipynb` and save a new copy.

### Your existing notebook environment

From a cloned checkout, install the project in the **Python environment your notebook kernel uses**:

```sh
python -m pip install -e ".[notebook]"
```

Restart the kernel, then:

```python
from proofread import Report

report = Report.load("/path/to/TabLint/demo/video/auto_mpg_dirty.proofread.json")
review = report.widget()
review
```

For your own data (live inference):

```python
import pandas as pd
import proofread

df = pd.read_csv("data.csv")
report = df.tablint.check(categorical=True)
review = report.widget()
review
```

Use `df.tablint.view(categorical=True)` to check and display in one step. `report.issues` gives the issue table; `report.highlight()` gives a pandas Styler. Add `label="outcome"` to check a target column. The original `df.proofread` accessor remains supported. `%load_ext proofread` registers both `%tablint df` and the original `%proofread df` magic.

The [JupyterLab startup guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html) and [uv's notebook guide](https://docs.astral.sh/uv/guides/integration/jupyter/) cover alternate server/kernel setups.

## VS Code

Prerequisites: Node.js 22 or newer, npm, and VS Code 1.90+ with the `code` command on PATH.

```sh
git clone https://github.com/James-Begin/TabLint.git && cd TabLint && npm run demo:vscode
```

The script uses `npm ci` with the extension's lockfile, builds a VSIX, installs it into your existing VS Code profile, and opens a light-themed demo workspace. It copies the saved Auto MPG report and CSV into `demo/.workspace/` on first use. Subsequent runs preserve your demo edits and ledger. No inference runs in this setup.

If VS Code requests Workspace Trust, inspect the local source and choose whether to trust it. After installation, an already running window may need **Developer: Reload Window** to activate the updated extension.

Hover the Civic's weight on CSV line 183, then inspect the Rabbit's horsepower on line 177. Quick Fix (`Ctrl+.` / `Cmd+.`) offers replace, flag and dismiss. Open **TabLint: Open decision ledger** in the Command Palette to inspect and undo decisions. The saved report includes numerical and origin-label issues; categorical-column checks are enabled in new live checks by default.

To restart from the original dataset, close the demo workspace, remove the disposable `demo/.workspace/` directory, and rerun `npm run demo:vscode`. This also removes its local decision history.

## Manual extension installation

For a ready-built extension plus saved data, [download the demo kit](../demo/showcase/TabLint-demo-kit.zip) and extract it. In VS Code run **Extensions: Install from VSIX…**, choose the included `tablint.vsix`, and open the extracted folder and CSV. This requires only VS Code, with no Node.js or Python setup.

To build without installing or opening VS Code:

```sh
npm run demo:vscode -- --prepare-only
```

Then use **Extensions: Install from VSIX…** and select `dist/tablint.vsix`, or run:

```sh
code --install-extension dist/tablint.vsix
code demo/.workspace/TabLint-demo.code-workspace
```

For saved replay, open a CSV beside its matching `name.proofread.json` file. **TabLint: Reload saved report** refreshes the diagnostics. The report format and import package still use the original `proofread` identifier for compatibility.

## Live CSV analysis

Install [uv](https://docs.astral.sh/uv/getting-started/installation/); it manages the required Python 3.12 interpreter and environment.

```sh
uv sync --locked
uv run tablint check demo/video/auto_mpg_dirty.csv --no-categorical --out reports/auto_mpg
uv run tablint view reports/auto_mpg.json
```

Omit `--no-categorical` to include low-cardinality categorical columns. Add `--label outcome` when checking a known label column. `--fast` selects the lower-latency TabPFN-3.5-Fast checkpoint. `--threshold 3` raises the numerical flag threshold.

First inference downloads TabPFN weights and may open the model-license acceptance flow. Accept that license yourself if you agree. Weights are cached separately and are not included in this repository. A GPU is optional; CPU inference can take several minutes. No API key is required for the saved demos.

To run a live check from VS Code, open this repository as the workspace and use **TabLint: Check this CSV with TabPFN-3.5**. For another workspace, set `proofread.command` to an absolute installed CLI command or `uv run --project /path/to/TabLint tablint`. See the [extension settings](../vscode-proofread/README.md#settings).

## Browser viewer

```sh
uv run --extra demo tablint app
```

Select a stored example to inspect without model inference. Uploading a CSV and pressing **Check with TabLint** performs a live check and requires the weights. The separate `demo/app.py` is a research audit viewer; the TabLint command launches `demo/proofread_app.py`.

## Terminal viewer

```sh
uv run tablint view demo/video/auto_mpg_dirty.proofread.json
```

`n` / `p`: next / previous issue; `a`: accept; `d`: dismiss; `u`: undo; `f`: flagged rows; `s`: save a cleaned CSV and decisions JSON. Open a saved report for replay or a CSV for live inference.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Widget is blank or `anywidget` cannot be imported | Select the Python kernel from the environment with `.[notebook]` installed, restart it, and rerun the widget cell. GitHub renders notebook previews statically; run locally for interaction. |
| Port 8888 is busy | Jupyter normally chooses the next free port; follow the printed URL or pass `--port 8890`. |
| Python version mismatch | Use the `uv` launcher to prepare Python 3.12, or select a Python 3.12 environment for manual installation. |
| `code` is missing | In VS Code on macOS, run **Shell Command: Install 'code' command in PATH**. On Windows/Linux, ensure the VS Code CLI is on PATH. Restart your terminal, then `npm run doctor`. |
| Node version error | Install Node.js 22+ and retry. |
| No squiggles | Confirm the matching `.proofread.json` is beside the CSV, reload the extension window, then **TabLint: Reload saved report**. Modified cells intentionally stop showing stale diagnostics. |
| Model access/download error | Complete the upstream model-license flow for live inference, or use a saved report to replay immediately. |
| Slow analysis | Replay saved reports, use a CUDA GPU, or try `--fast`. Larger tables and more checked columns increase work. |
| Categorical output differs from the video | The specific USA → Japan video suggestion is illustrative. The saved Auto MPG report includes numerical and origin-label issues; live categorical-column checks use their actual classifier results. |

[Back to the README](../README.md)
