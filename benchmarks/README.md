# Benchmark reproduction

The submission graphs use stored measurements; rebuilding them does not run a model:

```sh
uv run --with matplotlib==3.11.2 python benchmarks/make_showcase_figures.py
```

Run commands from the repository root. [Methodology and graphs](../docs/BENCHMARKS.md), [scope amendment](../docs/BENCHMARK_AMENDMENT.md) and [result directory guide](../results/README.md) describe the evidence.

## Fresh inference

These scripts use the published CUDA configurations, require TabPFN model access and download public datasets. Install the baseline extra only for the addendum. Use new directories to avoid overwriting the stored submission evidence; runners skip existing files.

```sh
uv sync --locked --extra benchmark --extra baselines
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.proofread_confirm --out results/reproduction/numerical
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.categorical_confirm --out results/reproduction/categorical
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.proofread_addendum --out results/reproduction/addendum
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.version_compare --out results/reproduction/versions
```

Consult each runner’s `--help` for dataset and seed selection. The numerical loader in `common.py` preserves the original numeric-column selection, finite-row filtering, binary class codes, deterministic 3,000-row cap and label-baseline settings.

## Summaries and figures

```sh
uv run python benchmarks/summarize_proofread.py results/reproduction/numerical
uv run python benchmarks/summarize_categorical.py results/reproduction/categorical
uv run python benchmarks/summarize_addendum.py results/reproduction/addendum
uv run python benchmarks/summarize_versions.py results/reproduction/versions
```

Summary scripts also regenerate their published documents where implemented; review the diff before committing. `make_proofread_results.py` rebuilds the numerical report. `make_showcase_figures.py` validates the 60-run numerical and 50-run categorical inventories before rebuilding all submission graphs.

## Video fixtures

`video_data.py` and `make_video_demos.py` prepare the canonical Auto MPG data and reports in [demo/video/](../demo/video/README.md). These support production rehearsals and UI checks. User setup starts in [Jupyter](../docs/GETTING_STARTED.md).

## CPU retail capability experiment

The [retail experiment](../docs/RETAIL_CPU_BENCHMARK.md) compares local TabPFN checkpoints on short product descriptions, high-cardinality stock/customer identifiers and larger historical contexts. It is separate from the published small-table suite and the current TabLint context selector. No inference API is involved.

```sh
uv run --with openpyxl python -m benchmarks.retail_cpu --prepare
uv run python -m benchmarks.retail_cpu --download
uv run python -m benchmarks.retail_cpu --sizes 1000 --out results/reproduction/retail_cpu
uv run --with matplotlib==3.11.2 python -m benchmarks.summarize_retail --root results/reproduction/retail_cpu
```

The summarizer writes the retail report and figure; use `--report` and `--figure` to choose alternative output paths. See the runner's `--help` and the frozen configuration for context sizes, ablations, seeds and per-worker resource limits. Raw input stays in `data_cache/`; structured output distinguishes completed, unsupported and resource-limited attempts.
