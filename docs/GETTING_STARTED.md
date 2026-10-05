# Getting started with TabLint

## VS Code: saved demo in one command

Prerequisites: Node.js 22 or newer, npm, and VS Code 1.90+ with the `code` command on PATH.

```sh
git clone https://github.com/James-Begin/TabLint.git && cd TabLint && npm run demo
```

The script uses `npm ci` with the extension's lockfile, builds a VSIX, installs it into your existing VS Code profile, and opens a light-themed demo workspace. It copies the saved Auto MPG report and CSV into `demo/.workspace/` on first use. Subsequent runs preserve your demo edits and ledger. No inference runs in this setup.

If VS Code requests Workspace Trust, inspect the local source and choose whether to trust it. After installation, an already running window may need **Developer: Reload Window** to activate the updated extension.

Hover the Civic's weight on CSV line 183, then inspect the Rabbit's horsepower on line 177. Quick Fix (`Ctrl+.` / `Cmd+.`) offers replace, flag and dismiss. Open **TabLint: Open decision ledger** in the Command Palette to inspect and undo decisions. The saved report is numeric-only; categorical checks are enabled in new live checks by default.

To restart from the original dataset, close the demo workspace, remove the disposable `demo/.workspace/` directory, and rerun `npm run demo`. This also removes its local decision history.

## Manual extension installation

For a ready-built extension plus saved data, [download the demo kit](../demo/showcase/TabLint-demo-kit.zip) and extract it. In VS Code run **Extensions: Install from VSIX…**, choose the included `tablint.vsix`, and open the extracted folder and CSV. This requires only VS Code, with no Node.js or Python setup.

To build without installing or opening VS Code:

```sh
npm run demo -- --prepare-only
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

## Python and notebooks

```sh
uv sync --locked --extra notebook
```

```python
import pandas as pd
import proofread

report = proofread.Report.load("demo/video/auto_mpg_dirty.proofread.json")
report.widget()  # saved interactive replay

# For your own data, with model weights:
df = pd.read_csv("data.csv")
report = proofread.Proofreader().check(df, categorical=True)
report.issues
report.highlight()
```

The [example notebook](../examples/proofread_notebook.ipynb) uses a saved report. The pandas accessor supports `df.proofread.view()` and `df.proofread.check()`.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| `code` is missing | In VS Code on macOS, run **Shell Command: Install 'code' command in PATH**. On Windows/Linux, ensure the VS Code CLI is on PATH. Restart your terminal, then `npm run doctor`. |
| Node version error | Install Node.js 22+ and retry. |
| No squiggles | Confirm the matching `.proofread.json` is beside the CSV, reload the extension window, then **TabLint: Reload saved report**. Modified cells intentionally stop showing stale diagnostics. |
| Model access/download error | Complete the upstream model-license flow for live inference, or use a saved report to replay immediately. |
| Slow analysis | Replay saved reports, use a CUDA GPU, or try `--fast`. Larger tables and more checked columns increase work. |
| Categorical output differs from the video | The specific USA → Japan video suggestion is illustrative. The saved Auto MPG report is numeric-only; live checks use their actual classifier results. |

[Back to the README](../README.md)
