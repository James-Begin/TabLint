# Contributing

Start with [Getting started](docs/GETTING_STARTED.md). Keep the product name **TabLint**; Python imports, report schemas and setting IDs retain `proofread` for compatibility.

## Python

```sh
uv sync --locked --extra notebook --extra demo
uv run python scripts/verify_submission.py
uv run pytest -q tests
uv run --extra notebook python scripts/rehearse_notebook.py --prepare-only
```

Weight-dependent inference tests skip when TabPFN-3.5 weights are not cached. Unit tests and saved-report checks do not require model-license acceptance. To reproduce inference tests, first obtain the weights through the upstream model-access flow. Research reproduction commands and limitations are in [the research overview](docs/RESEARCH_OVERVIEW.md).

## VS Code

```sh
cd vscode-proofread
npm ci
npm test
npm run test:integration
```

The integration test downloads VS Code into its test cache and uses a temporary profile. It covers diagnostics, categorical fixes, ledger persistence and undo. Linux needs a display; use `xvfb-run -a npm run test:integration` in headless environments. `npm run rehearse:vscode -- --prepare-only` from the repository root verifies the packaging path without modifying your installed extensions.

## Notebook JavaScript

```sh
cd tests/js
npm ci
npm test
```

## Pull requests and issues

Explain the concrete problem, resulting behavior and checks you ran. For bugs, include the interface, environment and a small anonymized reproducer; for inference results, include the model version and distinguish stored replay from a live run. Do not commit credentials, model weights, environments, build products or personal CSVs. Generated benchmark artifacts already in this repository are retained for reproducibility.
