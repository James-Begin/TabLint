# TabLint: technical overview

TabLint checks whether a recorded value fits the rest of its row. The primary workflow is [Jupyter and pandas](GETTING_STARTED.md#jupyter-notebooks); the same Python engine supports a CLI, terminal viewer, browser app and [VS Code integration](../vscode-proofread/README.md).

## Inference and review

[The engine](../proofread/core.py) runs TabPFN-3.5 locally, on CUDA when available or on CPU. It does not call the TabPFN API. First inference downloads the model weights through Prior Labs’ model-access flow; weights are not bundled here.

For each numerical column, the engine predicts that column from the other columns, including a designated label when supplied. Five-fold predictions hold the row being checked out of that fold’s context. TabPFN’s distribution gives a median, a 10th–90th percentile interval, and the recorded value’s cumulative probability. The score is −log₁₀ of its two-sided tail probability, clipped for numerical stability.

Categorical cells and optional labels use an out-of-fold classifier. The recorded class receives a surprise score based on its predicted probability, and a more likely class can be suggested. Category selection and preprocessing are implemented in `Proofreader.categorical_columns` and `categorical_scores`; high-cardinality identifiers and free text are excluded.

Common numeric slips and rare text typos receive possible-cause hints. Related-column values provide row context, not causal attribution. Flags and suggested replacements require source review.

The [notebook widget](../proofread/notebook.py) returns a cleaned copy and current decision snapshot to pandas. The optional editor integration maintains a persistent, append-only ledger with reasons, flag-for-review decisions and undo protection against later manual edits. Saved JSON reports can be reopened without model inference.

## Evidence

| Question | Protocol and complete results |
| --- | --- |
| Detection and correction of injected numerical errors; label errors | [Numerical results](PROOFREAD_RESULTS.md) |
| Comparison with the optional TabPFN unsupervised baseline; threshold analysis | [Baseline comparison](COMPARISON_TABPFN_EXTENSIONS.md) |
| Detection of plausible categorical swaps | [Categorical results](CATEGORICAL_RESULTS.md) |
| TabPFN-3.5, Fast, v3 and v2 comparison | [Model comparison](VERSION_COMPARISON.md) |
| Findings on familiar, unmodified tables, including unverified flags and missed Iris errors | [Exploratory gallery](FAMOUS_DATASETS.md) |

The [benchmark showcase](BENCHMARKS.md) presents the selected 12-dataset numerical view and categorical results.

## Reproduce

From the repository root:

```sh
uv sync --locked --extra notebook --extra demo
uv run pytest -q tests
# Rebuild the showcase graphs from stored measurements; no inference.
uv run --with matplotlib==3.11.2 python benchmarks/make_showcase_figures.py
```

Fresh numerical confirmation runs require TabPFN weights, public-data downloads and a CUDA GPU under the published configuration:

```sh
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.proofread_confirm --out results/reproduction/numerical
uv run python benchmarks/summarize_proofread.py results/reproduction/numerical
```

Use a separate output directory: runners skip existing run files. See [benchmarks/](../benchmarks/README.md) for categorical, model-version and addendum commands. These are expensive inference experiments; graph generation and most product tests use stored reports or mocks.

## Limitations

- Synthetic corruptions and known review budgets do not establish accuracy on a new table. Precision@k differs from precision at the product’s default threshold.
- Predictive surprise is not a calibrated probability that a value is wrong. Natural exceptions can be flagged; small plausible errors can be missed.
- Independent columns give little predictive context. Sparse categories and high-cardinality text may have too little support; identifiers and free text are not checked as categories.
- Runtime grows with columns and folds. CPU inference can take several minutes; the [model comparison](VERSION_COMPARISON.md) documents the tested Fast checkpoint.
- The editor supports comma-separated CSVs with one record per line. Notebook and terminal review semantics differ from the persistent editor ledger; see [setup and export](GETTING_STARTED.md).
- The optional `tabpfn-extensions` benchmark baseline required a float64 workaround on CUDA. The [bug report](bugs/TABPFN_EXTENSIONS_CUDA_DTYPE.md) and [known issues](KNOWN_ISSUES.md) document the effect and validation limits. TabLint’s normal scoring uses a different path.

## Data and licensing

Project code is [Apache-2.0](../LICENSE). Dependencies, model weights, datasets and fonts retain their own licenses. Public benchmark tables are fetched by dataset ID; the included Auto MPG showcase tables have [CC BY 4.0 attribution and an injection answer key](../demo/video/README.md). See [NOTICE](../NOTICE).
