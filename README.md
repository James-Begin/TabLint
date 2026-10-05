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
  <a href="#set-up-in-jupyter"><b>Set up in Jupyter</b></a> · <a href="#demo"><b>Watch the demo</b></a> · <a href="#how-it-works"><b>How it works</b></a> · <a href="docs/PROOFREAD_RESULTS.md"><b>Benchmarks</b></a>
</p>

TabLint finds values that look ordinary in a column but suspicious **for their row**. It uses TabPFN-3.5 to check numerical cells, categorical values and labels, explains each flag, and helps you review a suggested correction.

Review suspicious values in your own pandas DataFrame, inspect TabPFN’s reasoning, and return a cleaned table without leaving Jupyter.

## Demo

[![Watch the TabLint product showcase](demo/showcase/preview.gif)](https://github.com/James-Begin/TabLint/blob/main/demo/showcase/TabLint-demo.mp4)

**[Watch the full 1:28 showcase](https://github.com/James-Begin/TabLint/raw/refs/heads/main/demo/showcase/TabLint-demo.mp4)**

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

Click a highlighted cell to inspect its expected value, plausible range and reason. **Accept**, **dismiss** or **undo** a suggestion. After reviewing, run another cell to get your results:

```python
cleaned = review.cleaned
changes = review.decisions
```

The original DataFrame stays intact. First analysis downloads TabPFN weights and requires first-use model-license acceptance. A GPU is optional; CPU inference can take several minutes. [Setup, export and troubleshooting →](docs/GETTING_STARTED.md)

### Need a new Jupyter environment?

Install **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, then:

```sh
git clone https://github.com/James-Begin/TabLint.git && cd TabLint && uv run --extra notebook jupyter lab
```

This prepares Python 3.12 and the notebook dependencies, then opens JupyterLab. Create a Python notebook and use the code above with your CSV.

### Other integrations

**VS Code:** install the extension, open your CSV and run **TabLint: Check this CSV with TabPFN-3.5**. Includes CSV squiggles, Quick Fix actions and a persistent decision ledger. [Editor setup →](docs/GETTING_STARTED.md#vs-code)

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

In the pre-registered injected-error benchmarks, TabLint beat the best baseline selected separately for each dataset on **13/14 numerical datasets** and **10/10 categorical datasets**. The tests use synthetic corruptions on public tables; these results do not guarantee performance on a new dataset.

[Numeric results](docs/PROOFREAD_RESULTS.md) · [Categorical results](docs/CATEGORICAL_RESULTS.md) · [Pre-registrations](docs/PREREGISTRATIONS.md) · [Model comparison](docs/VERSION_COMPARISON.md)

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

Built for the TabPFN hackathon. The original research modules remain in `chainofcustody/` and `auditkit/`; they are documented separately in [the research overview](docs/RESEARCH_OVERVIEW.md).

## License

Code: **[Apache-2.0](LICENSE)**. [Auto MPG demo data](demo/video/README.md): UCI, CC BY 4.0. TabPFN weights, dependencies and bundled fonts retain their own licenses; weights are not redistributed. [Notices](NOTICE).
