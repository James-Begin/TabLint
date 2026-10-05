# Benchmark reproduction

The submission graphs use stored measurements; rebuilding them does not run a model:

```sh
uv run --with matplotlib==3.11.2 python benchmarks/make_showcase_figures.py
```

Run commands from the repository root. [Methodology and graphs](../docs/BENCHMARKS.md), [frozen protocols](../docs/PREREGISTRATIONS.md) and [result directory guide](../results/README.md) describe the evidence.

## Fresh inference

These scripts use the published CUDA configurations, require TabPFN model access and download public datasets. Install the baseline extra only for the addendum. Use new directories to avoid overwriting the stored submission evidence; runners skip existing files.

```sh
uv sync --locked --extra baselines
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

Summary scripts also regenerate their published documents where implemented; review the diff before committing. `make_proofread_results.py`, `make_figures.py` and `make_version_figure.py` rebuild complete-suite reports and historical technical figures. `make_showcase_figures.py` verifies the original run inventory, creates the selected 12-row numerical view and recalculates its statistics.

## Real tables and production fixtures

`famous_datasets.py` and `make_famous_gallery.py` produce the exploratory gallery, with source checks that distinguish verified errors from unusual or unverified values. `real_pima.py` checks known impossible-zero patterns.

`make_demos.py`, `video_data.py`, `make_video_demos.py` and `make_tui_screenshot.py` prepare stored report fixtures and production assets. They support development and the video; user setup starts in [Jupyter](../docs/GETTING_STARTED.md).
