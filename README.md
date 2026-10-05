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
  <a href="#try-it-in-jupyter"><b>Try it in Jupyter</b></a> · <a href="#demo"><b>Notebook demo</b></a> · <a href="#how-it-works"><b>How it works</b></a> · <a href="docs/PROOFREAD_RESULTS.md"><b>Benchmarks</b></a>
</p>

TabLint finds values that look ordinary in a column but suspicious **for their row**. It uses TabPFN-3.5 to check numerical cells, categorical values and labels, explains each flag, and helps you review a suggested correction.

A Honda Civic with four cylinders and 53 horsepower is recorded at **4,354 lb**. The saved demo report expects about **1,877 lb**, with an 80% interval of **1,720–2,034 lb**. TabLint brings that evidence into your Jupyter notebook.

## Demo

![TabLint reviewing a saved report in JupyterLab](docs/figures/tablint_jupyter.png)

**[Browse the demo notebook](examples/tablint_demo.ipynb)** · [Sample data and saved predictions](demo/video/) · [Notebook guide](docs/GETTING_STARTED.md#jupyter-notebooks)

## Try it in Jupyter

Install **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, then run:

```sh
git clone https://github.com/James-Begin/TabLint.git && cd TabLint && uv run --extra notebook tablint demo
```

One command prepares Python 3.12, JupyterLab and the widget, then opens an editable demo notebook. Choose **Run → Run All Cells**. **No model download, GPU or API key is needed for the saved replay.** No Node.js or editor extension is required.

1. Click a highlighted cell to inspect the expected value, plausible range and reason.
2. **Accept**, **dismiss** or **undo** a suggestion inside the table.
3. Get `review.cleaned` and `review.decisions` back as pandas DataFrames; export them when ready.

The notebook runs on saved Auto MPG numerical and origin-label predictions. Live analysis also handles low-cardinality categorical columns. The launcher preserves your working notebook in `demo/notebook-workspace/`. [Setup and troubleshooting →](docs/GETTING_STARTED.md)

### Your own DataFrame

In a notebook using the same environment:

```python
import pandas as pd
import proofread  # registers df.tablint; original imports stay compatible

df = pd.read_csv("data.csv")
review = df.tablint.view(categorical=True)
review
```

After reviewing, run another cell:

```python
cleaned = review.cleaned
changes = review.decisions
```

Live checks need TabPFN weights and first-use model-license acceptance. A GPU is optional; CPU inference can take several minutes. For existing Jupyter environments, install the notebook extra and select that Python kernel: [notebook setup](docs/GETTING_STARTED.md#your-existing-notebook-environment).

### Other integrations

**VS Code:** `npm run demo:vscode` opens the saved CSV demo (Node.js 22+ and VS Code required). It includes CSV squiggles, Quick Fix actions and a persistent decision ledger. The [editor demo kit](demo/showcase/TabLint-demo-kit.zip) includes a ready-built VSIX. [Editor setup →](docs/GETTING_STARTED.md#vs-code)

**Terminal:** `uv run tablint check data.csv --out reports/data`, then `uv run tablint view reports/data.json`.

**Browser viewer:** `uv run --extra demo tablint app`.

### Editor video showcase

[![Watch the TabLint editor showcase](demo/showcase/preview.gif)](https://github.com/James-Begin/TabLint/blob/main/demo/showcase/TabLint-demo.mp4)

**[Watch / download the full 1:28 showcase](https://github.com/James-Begin/TabLint/raw/refs/heads/main/demo/showcase/TabLint-demo.mp4)** · [Editable animation](showcase/)

This animation shows the VS Code integration. Numerical examples use saved inference results; the specific USA → Japan suggestion is illustrative. [Provenance and review](demo/showcase/README.md).

## Features

| Feature | What you get |
| --- | --- |
| **Jupyter + pandas** | Interactive cell review, accept/dismiss/undo, a cleaned DataFrame and a decision snapshot in Python. |
| **Numerical checks** | Row-specific expectations, 80% plausible ranges, and hints for decimal slips, swapped digits, sign flips and missing-value zeros. |
| **Categorical checks** | Low-cardinality categories scored against the rest of the row, with likely alternatives and possible typo hints. |
| **Optional VS Code integration** | CSV squiggles, hover explanations, Quick Fix actions and a Problems list. |
| **Editor decision ledger** | The VS Code integration stores before/after values, reasons, timestamps and review notes; undo protects newer manual edits. |
| **More ways to work** | Terminal viewer, browser viewer, CLI, Python API and MCP tools. |
| **Replayable reports** | JSON, highlighted HTML, Markdown and an issues CSV, with saved demos for a quick first run. |

## How it works

For each column, TabLint asks TabPFN to predict a cell from the other columns. It uses **out-of-fold predictions**, so the row being checked is held out of that fold's context.

- **Numbers:** score the recorded value against TabPFN's predictive distribution; show the median and 10th–90th percentiles. Test common slips for a possible explanation.
- **Categories and labels:** use the classifier's probability for the recorded class, then suggest a more likely class for review.
- **Decisions:** review in the notebook, then retrieve the cleaned table and current decisions in Python. The optional VS Code integration also offers flag-for-review actions and a persistent ledger.

A flag means **check the source record**, not a confirmed error. Surprise scores rank anomalies; they are not calibrated probabilities that a value is wrong. [Implementation](proofread/core.py) · [Technical and research overview](docs/RESEARCH_OVERVIEW.md)

## Evidence

In the pre-registered injected-error benchmarks, TabLint beat the best baseline selected separately for each dataset on **13/14 numerical datasets** and **10/10 categorical datasets**. The tests use synthetic corruptions on public tables; these results do not guarantee performance on a new dataset.

[Numeric results](docs/PROOFREAD_RESULTS.md) · [Categorical results](docs/CATEGORICAL_RESULTS.md) · [Pre-registrations](docs/PREREGISTRATIONS.md) · [Model comparison](docs/VERSION_COMPARISON.md)

Independent columns provide little context for this approach; high-cardinality identifiers and free text are not treated as categories. Live inference cost grows with the columns and folds. The extension supports comma-separated CSVs with one record per line. [Limitations and reproducibility](docs/RESEARCH_OVERVIEW.md#limitations).

## Project guide

| Start here | Contents |
| --- | --- |
| [Getting started](docs/GETTING_STARTED.md) | Setup, saved replay, live checks and troubleshooting |
| [Jupyter demo](examples/tablint_demo.ipynb) | Interactive review and cleaned pandas DataFrames |
| [VS Code extension](vscode-proofread/) | Optional editor workflow and persistent ledger |
| [Python engine](proofread/) | Inference, reports, terminal and notebook interfaces |
| [Demo](demo/video/) / [showcase](showcase/) | Attributed data, approved video and animation source |
| [Benchmarks](benchmarks/) / [results](results/) | Reproduction scripts and measured artifacts |
| [Contributing](CONTRIBUTING.md) | Development and verification commands |

Built for the TabPFN hackathon. The original research modules remain in `chainofcustody/` and `auditkit/`; they are documented separately in [the research overview](docs/RESEARCH_OVERVIEW.md).

## License

Code: **[Apache-2.0](LICENSE)**. [Auto MPG demo data](demo/video/README.md): UCI, CC BY 4.0. TabPFN weights, dependencies and bundled fonts retain their own licenses; weights are not redistributed. [Notices](NOTICE).
