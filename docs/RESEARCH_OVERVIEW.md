# TabLint: technical overview

TabLint checks whether a recorded value fits the rest of its row. The primary workflow is [Jupyter and pandas](GETTING_STARTED.md#jupyter-notebooks); the same Python engine supports a CLI, terminal viewer, browser app and [VS Code integration](../vscode-proofread/README.md).

## Inference and review

[The engine](../proofread/core.py) runs TabPFN-3.5 locally, on CUDA when available or on CPU. It does not call the TabPFN API. First inference downloads the model weights through Prior Labs’ model-access flow; weights are not bundled here.

For each numerical column, the engine predicts that column from the other columns, including a designated label when supplied. Five-fold predictions hold the row being checked out of that fold’s context. TabPFN’s distribution gives a median, a 10th–90th percentile interval, and the recorded value’s cumulative probability. The score is −log₁₀ of its two-sided tail probability, clipped for numerical stability.

Categorical cells and optional labels use an out-of-fold classifier. The recorded class receives a surprise score based on its predicted probability, and a more likely class can be suggested. `Proofreader.categorical_columns` selects low-cardinality check targets. The separate `categorical_context_columns` retains every string/object, boolean and declared pandas category input, including high-cardinality IDs and descriptions. Observed values are encoded in sorted order, missing values remain NaN, and categorical feature indices are supplied to each model. The numeric-code categorical cap is set to at least the table row count so TabPFN does not silently reclassify declared IDs as numerical. Numerical identifiers can be declared with pandas `category` dtype. Codes represent category identities, not semantic text embeddings. Selecting `columns` limits the check targets while preserving the other input columns as context; `categorical=False` disables categorical-cell checks, not categorical context.

Common numeric slips and rare text typos receive possible-cause hints. Related-column values provide row context, not causal attribution. Flags and suggested replacements require source review.

The [notebook widget](../proofread/notebook.py) returns a cleaned copy and current decision snapshot to pandas. The optional editor integration maintains a persistent, append-only ledger with reasons, flag-for-review decisions and undo protection against later manual edits. Saved JSON reports can be reopened without model inference.

## Evidence

| Question | Protocol and complete results |
| --- | --- |
| Detection and correction of injected numerical errors; label errors | [Numerical results](PROOFREAD_RESULTS.md) |
| Comparison with the optional TabPFN unsupervised baseline; threshold analysis | [Baseline comparison](COMPARISON_TABPFN_EXTENSIONS.md) |
| Detection of plausible categorical swaps | [Categorical results](CATEGORICAL_RESULTS.md) |
| TabPFN-3.5, Fast, v3 and v2 comparison | [Model comparison](VERSION_COMPARISON.md) |
| v3 versus 3.5 with and without added high-cardinality context on CPU | [High-cardinality study](HIGH_CARDINALITY_RESULTS.md) |

The [benchmark showcase](BENCHMARKS.md) presents all 12 retained numerical datasets and categorical results. Two dataset executions were excluded after reported test errors; [the amendment](BENCHMARK_AMENDMENT.md) documents the scope correction. Stored measurements are in [results/](../results/README.md). The categorical confirmation runner preserves its [original context policy](../benchmarks/categorical_reference.py); those 50 stored runs are not measurements of the newly expanded product context.

### High-cardinality context: methods and findings

The supplied experiment checks numerical errors on employee salaries and medical charges (1,000 sampled rows each) and Auto MPG (392 complete rows). It reuses the numerical injection policy, holds checked rows out of five-fold model contexts, and compares TabPFN v3 and 3.5 in paired arms with and without sorted identity codes for additional context. Five seeds produce 15 tables and 60 arm records. No run is excluded.

With added context, 3.5 reaches **81.3% mean precision@k versus 79.1% for v3**: **+2.19 points**, nine wins, two losses and four ties; one-sided Wilcoxon p = 0.0122 over table pairs. The difference in context benefit between versions is **+1.66 points** (p = 0.0287). Without the added context, the version gap is **+0.53 points** (p = 0.3242). These modest gains support retaining useful context in the tested configurations; repeated seeds across only three source datasets limit generalisation. CPU time with context averages 156 seconds for 3.5 versus 80 for v3.

The stored study used TabPFN 9.0.0's default numeric-code category cap. Explicit indices alone do not keep vocabularies above that cap categorical. Its measurements test the retained coded context and are not a performance test of the updated product's cap override. Identity codes provide no semantic text embeddings; the study also supplies no 100k-row result. [Full protocol, per-dataset table, provenance and reproducibility](HIGH_CARDINALITY_RESULTS.md) keep this evidence separate from the earlier numerical, categorical and retail experiments.

## Reproduce

From the repository root:

```sh
uv sync --locked --extra notebook --extra demo --extra benchmark
uv run pytest -q tests
# Rebuild the showcase graphs from stored measurements; no inference.
uv run --with matplotlib==3.11.2 python benchmarks/make_showcase_figures.py
uv run --with matplotlib==3.11.2 python -m benchmarks.summarize_high_cardinality
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
- Independent columns give little predictive context. Sparse categories and unique identifiers may provide too little support. High-cardinality strings remain categorical context, but are not checked as categorical targets; their codes do not capture semantic text similarity.
- Runtime grows with columns and folds. CPU inference can take several minutes; the [model comparison](VERSION_COMPARISON.md) documents the tested Fast checkpoint.
- The editor supports comma-separated CSVs with one record per line. Notebook and terminal review semantics differ from the persistent editor ledger; see [setup and export](GETTING_STARTED.md).
- The optional `tabpfn-extensions` benchmark baseline required a float64 workaround on CUDA. The [bug report](bugs/TABPFN_EXTENSIONS_CUDA_DTYPE.md) documents the effect and validation limits. TabLint’s normal scoring uses a different path.

## Data and licensing

Project code is [Apache-2.0](../LICENSE). Dependencies, model weights, datasets and fonts retain their own licenses. Public benchmark tables are fetched by dataset ID; the included Auto MPG showcase tables have [CC BY 4.0 attribution and an injection answer key](../demo/video/README.md). See [NOTICE](../NOTICE).
