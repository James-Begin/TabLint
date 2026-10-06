<p align="center">
  <img src="docs/figures/tablint-banner.svg" alt="TabLint — spellcheck for tables, powered by TabPFN-3.5" width="100%">
</p>
<p align="center">
  <a href="https://github.com/James-Begin/TabLint/actions/workflows/ci.yml"><img src="https://github.com/James-Begin/TabLint/actions/workflows/ci.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-157F79" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/model-TabPFN%203.5-157F79" alt="TabPFN 3.5">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB" alt="Python 3.12">
</p>
<p align="center">
  <a href="#set-up-in-jupyter"><b>Set up in Jupyter</b></a> · <a href="#set-up-in-vs-code"><b>Set up in VS Code</b></a> · <a href="#demo"><b>Watch the demo</b></a> · <a href="#how-it-works"><b>How it works</b></a> · <a href="docs/BENCHMARKS.md"><b>Benchmarks</b></a>
</p>

TabLint finds values that look ordinary in a column but suspicious **for their row**. It uses TabPFN-3.5 to check numerical cells, categorical values and labels, explains each flag, and helps you review a suggested correction.

Review suspicious values in your own pandas DataFrame, inspect TabPFN’s reasoning, and return a cleaned table without leaving Jupyter.

## Demo

https://github.com/user-attachments/assets/8a850f7f-b17a-4e3a-82a1-bee73d5cfe09

**[Open or download the full MP4 showcase](https://github.com/James-Begin/TabLint/raw/refs/heads/main/demo/showcase/TabLint-demo.mp4)** · 1:28 · 1080p · 30 fps with audio

The video illustrates the editor integration. Numerical examples use saved inference results; the specific USA → Japan suggestion is illustrative. [Video provenance](demo/showcase/README.md).

## Set up in Jupyter

### Already using a notebook?

Use a **Python 3.12** kernel with Git installed. Run this installation cell:

```python
%pip install "tabpfn-proofread[notebook] @ git+https://github.com/James-Begin/TabLint.git"
```

Restart the kernel, then check **your own data**:

```python
import pandas as pd
import proofread  # registers the TabLint pandas accessor

df = pd.read_csv("data.csv")
review = df.tablint.view(categorical=True)
review
```

![TabLint running in JupyterLab: flagged Auto MPG cells and the Honda Civic weight explanation](docs/figures/tablint_jupyter.png)

*Actual JupyterLab capture using Auto MPG data and saved TabPFN results: the Civic’s recorded weight is 4,354 lb, while the model expects about 1,877 lb.*

Click a highlighted cell to inspect its expected value, plausible range and reason. **Accept**, **dismiss** or **undo** a suggestion. After reviewing, run another cell to get your results:

```python
cleaned = review.cleaned
changes = review.decisions
```

<details>
<summary><b>See a reviewed fix returned to pandas</b></summary>

![JupyterLab returning the accepted 0.0 to 70 horsepower fix and its decision snapshot to pandas](docs/figures/tablint_jupyter_results.png)

The Rabbit’s accepted horsepower suggestion appears in `review.cleaned`; its original value, suggestion, reason and review state appear in `review.decisions`.

</details>

The original DataFrame stays intact. First analysis downloads TabPFN weights and requires first-use model-license acceptance. A GPU is optional; CPU inference can take several minutes. [Setup, export and troubleshooting →](docs/GETTING_STARTED.md)

### Need a new Jupyter environment?

Install **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, then:

```sh
git clone https://github.com/James-Begin/TabLint.git && cd TabLint && uv run --extra notebook jupyter lab
```

This prepares Python 3.12 and the notebook dependencies, then opens JupyterLab. Create a Python notebook and use the code above with your CSV.

## Set up in VS Code

Install **[uv](https://docs.astral.sh/uv/getting-started/installation/)** and **[VS Code](https://code.visualstudio.com/)**. Prepare the Python engine from a terminal (skip cloning if you already have the checkout):

```sh
git clone https://github.com/James-Begin/TabLint.git && cd TabLint && uv sync
```

1. **Install the extension:** download [TabLint’s VSIX](demo/showcase/tablint.vsix). In VS Code, open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`), run **Extensions: Install from VSIX…**, and select the downloaded file.
2. **Connect the engine:** open your data folder and set **Proofread: Command** in Settings to the command below, replacing the path with your TabLint checkout. If you open the TabLint repository itself as the workspace, the default command works.
3. **Check your CSV:** open your own CSV, then run **TabLint: Check this CSV with TabPFN-3.5** from the Command Palette. Choose **(no label column)** unless you also want to check a known target column. First inference uses the same model setup described above.

```json
{
  "proofread.command": "uv run --project /absolute/path/to/TabLint tablint"
}
```

<table>
  <tr>
    <th align="center">VS Code</th>
    <th align="center">Browser UI</th>
  </tr>
  <tr>
    <td align="center"><a href="docs/figures/tablint_vscode_compact.png"><img src="docs/figures/tablint_vscode_compact.png" width="360" alt="VS Code CSV data view with the flagged Civic weight, model explanation and Quick Fix action"></a></td>
    <td align="center"><a href="docs/figures/tablint_browser_compact.png"><img src="docs/figures/tablint_browser_compact.png" width="360" alt="Browser app with highlighted Auto MPG data and TabPFN’s suggested Civic weight, plausible range and explanation"></a></td>
  </tr>
</table>

*Actual UI captures using saved Auto MPG results. Click either screenshot to enlarge.*

In **VS Code**, hover a squiggle for the explanation and use Quick Fix (`Ctrl+.` / `Cmd+.`) to replace, flag for review or dismiss. The **browser UI** shows a spreadsheet-style table with highlighted cells and summary counts; select an issue to inspect its expected range and row context.

<details>
<summary><b>See the decision ledger after a fix</b></summary>

![VS Code decision ledger recording a real 0.0 to 70 horsepower fix, its reason and an Undo action](docs/figures/tablint_vscode_ledger.png)

Run **TabLint: Open decision ledger** to view changes, reasons and review notes, or undo a decision. The ledger persists beside your CSV.

</details>

[Build the extension from source, configure paths and troubleshoot →](docs/GETTING_STARTED.md#vs-code)

### More integrations

**Terminal:** `uv run tablint check data.csv --out reports/data`, then `uv run tablint view reports/data.json`.

**Browser viewer:** `uv run --extra demo tablint app`, then upload your CSV.

## Features

| Feature | What you get |
| --- | --- |
| **Jupyter + pandas** | Interactive cell review, accept/dismiss/undo, a cleaned DataFrame and a decision snapshot in Python. |
| **Numerical checks** | Row-specific expectations, 80% plausible ranges, and hints for decimal slips, swapped digits, sign flips and missing-value zeros. |
| **Categorical checks** | Low-cardinality categories scored against the rest of the row, with likely alternatives and possible typo hints. |
| **Optional VS Code integration** | CSV squiggles, hover explanations, Quick Fix actions and a Problems list. |
| **Editor decision ledger** | The VS Code integration stores before/after values, reasons, timestamps and review notes; undo protects newer manual edits. |
| **More ways to work** | Terminal viewer, browser viewer, CLI, Python API and MCP tools. |
| **Replayable reports** | Save your analysis as JSON, highlighted HTML, Markdown and an issues CSV; reopen reports for later review. |

## How it works

For each column, TabLint asks TabPFN to predict a cell from the other columns. It uses **out-of-fold predictions**, so the row being checked is held out of that fold's context.

- **Numbers:** score the recorded value against TabPFN's predictive distribution; show the median and 10th–90th percentiles. Test common slips for a possible explanation.
- **Categories and labels:** use the classifier's probability for the recorded class, then suggest a more likely class for review.
- **Decisions:** review in the notebook, then retrieve the cleaned table and current decisions in Python. The optional VS Code integration also offers flag-for-review actions and a persistent ledger.

A flag means **check the source record**, not a confirmed error. Surprise scores rank anomalies; they are not calibrated probabilities that a value is wrong. [Implementation](proofread/core.py) · [Technical and research overview](docs/RESEARCH_OVERVIEW.md)

## Evidence

In the retained numerical benchmark, TabLint beat each dataset's strongest tested alternative on **12/12 datasets**, averaging **+22.2 percentage points** in precision@k. In the categorical confirmation experiment, it won on **10/10 datasets**, averaging **+6.2 points**.

![TabLint numerical benchmark: twelve selected datasets, twelve wins and a 22.2 percentage-point mean precision gain versus the strongest tested alternative](docs/figures/tablint_benchmark_numerical.png)

Precision@k is the share of the top k flagged cells that are injected errors, with k equal to the number of injected errors. Two numerical dataset executions were excluded after reported test errors; [the amendment](docs/BENCHMARK_AMENDMENT.md) documents the revised 12-dataset scope. These tests use synthetic corruptions on public tables; they do not guarantee performance on a new dataset.

[Benchmark graphs and methodology](docs/BENCHMARKS.md) · [Numeric results](docs/PROOFREAD_RESULTS.md) · [Categorical results](docs/CATEGORICAL_RESULTS.md) · [Scope amendment](docs/BENCHMARK_AMENDMENT.md) · [Model comparison](docs/VERSION_COMPARISON.md)

During baseline benchmarking, we documented a CUDA float32/float64 crash in `tabpfn-extensions`. The [bug report](docs/bugs/TABPFN_EXTENSIONS_CUDA_DTYPE.md) includes reproduction steps, affected versions, the benchmark workaround and a proposed upstream fix. It affects the optional extension baseline; TabLint's normal inference uses a different scoring path.

Independent columns provide little context for this approach; high-cardinality identifiers and free text are not treated as categories. Live inference cost grows with the columns and folds. The extension supports comma-separated CSVs with one record per line. [Limitations and reproducibility](docs/RESEARCH_OVERVIEW.md#limitations).

## Project guide

| Start here | Contents |
| --- | --- |
| [Getting started](docs/GETTING_STARTED.md) | Install, check your data, export and troubleshoot |
| [Notebook integration](proofread/notebook.py) | Interactive review and cleaned pandas DataFrames |
| [VS Code extension](vscode-proofread/) | Optional editor workflow and persistent ledger |
| [Python engine](proofread/) | Inference, reports, terminal and notebook interfaces |
| [Showcase](showcase/) | Video production source and attributed example data |
| [Benchmarks](benchmarks/) / [results](results/) | Reproduction scripts and measured artifacts |
| [Contributing](CONTRIBUTING.md) | Development and verification commands |

Built for the TabPFN hackathon. The repository contains the product, its tests, reproducible benchmark evidence and the editable showcase source.

## License

Code: **[Apache-2.0](LICENSE)**. [Auto MPG demo data](demo/video/README.md): UCI, CC BY 4.0. TabPFN weights, dependencies and bundled fonts retain their own licenses; weights are not redistributed. [Notices](NOTICE).
