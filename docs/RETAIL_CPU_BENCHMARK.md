# Retail text and identifier benchmark

This is a new **CPU feasibility experiment**, separate from the completed small-table benchmark. It tests TabPFN directly; TabLint's current context selector does not yet pass free-text descriptions or high-cardinality identifiers through to the model. It must not be presented as an existing TabLint feature or a confirmed TabPFN-3.5 advantage.

## Dataset and task

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), contributed by Daqing Chen under CC BY 4.0, contains 1,067,371 transaction rows. After removing cancellations, nonpositive prices/quantities, missing descriptions/customers, and exact duplicates, 779,425 rows remain. There are 4,631 stock codes, 5,878 customer IDs and 5,241 descriptions. Descriptions have a median length of 27 characters: these are short product names, not long prose.

Predict log unit price from stock code, description, quantity, invoice date, customer ID and country. Train on transactions before 1 August 2011 and test on later transactions; remove any invoice spanning the boundary. Sample nested contexts of 1,000, 5,000, 10,000 and **100,001 training rows**, with the same 1,024 held-out rows for each seed. Invoice numbers are used only for split validation and never as a feature. Test categories are learned from training data only; unseen identifiers become missing values.

Inject errors into 3% of held-out prices, cycling through ×10, ÷10 and a price swap differing by at least 2×. Original observed prices are references for these artificial errors; they are not independently audited truth. Training references remain uncorrupted. This is a historical-context price-check experiment, not a repeat of TabLint's cross-fitted whole-table evaluation.

## Fair comparison

Compare local V2, V3, V3.5 Base and V3.5 Fast with one estimator, six CPU threads, the same current `tabpfn==9.1.0` text/date preprocessing, and no internal row subsampling. Description is a pandas string; stock code, customer and country are explicitly categorical. The current package encodes text using character n-grams and SVD; this does not test the API-only Plus text system. Version-specific remaining preprocessing defaults are retained and recorded where relevant.

A SKU median/MAD baseline uses the same training rows and falls back to global statistics for unseen SKUs. This is an essential comparator because retail prices often repeat by product. Full-input, no-description and no-identifier runs isolate which inputs help. Report precision at the known injected-error count, AUROC, clean price/log-price error, injected-price correction error, interval coverage, runtime and peak memory. A single seed is only a feasibility pilot; three seeds are specified for an expanded experiment. All versions must use matching completed runs to support paired claims.

V3 already supports large tables and text-related tasks; 100k rows alone does not establish a new 3.5 capability. Measure a benefit rather than assume one. See [the V3 report](https://priorlabs.ai/technical-reports/tabpfn-3) and [V3.5 report](https://priorlabs.ai/technical-reports/tabpfn-3-5).

## Run locally

```bash
# From the repository; no GPU or inference API required.
uv sync
uv run --with openpyxl python -m benchmarks.retail_cpu --prepare
uv run python -m benchmarks.retail_cpu --download
uv run python -m benchmarks.retail_cpu --sizes 1000

# Expand only after inspecting runtime and memory.
uv run python -m benchmarks.retail_cpu --sizes 5000,10000,100001
uv run python -m benchmarks.retail_cpu --sizes 1000 --seeds 1101,1102,1103 --ablations full,no_text,no_ids
```

Dataset and model downloads require network access; inference is local CPU work. The existing accepted Prior Labs model license is needed for V3/3.5 downloads. Raw data and model weights stay in ignored caches.

[`retail_config.json`](../benchmarks/retail_config.json) fixes the dataset checksum, split, seeds, corruption and resource settings before inference. Each worker has a 240-second wall-time budget and an 8 GiB RSS limit, including its child processes; checkpoint downloads are done separately. Exceeding a limit records a timeout or memory failure, never a fabricated score. A CPU-only guard override permits testing above the recommended CPU size, while each checkpoint's actual sample limit remains enforced. Model `fit` and `predict` time are recorded separately from worker startup; wall time includes loading the already-cached model. Peak RSS includes the worker's dataframe and preprocessing.

The retained [`dataset_profile.json`](../results/retail_cpu/dataset_profile.json) records the source and prepared-table hashes, actual cardinalities and split sizes. Run records include configuration/split hashes, input dtypes, completed training size, expanded feature count, model sample limit, actual estimator count and SKU coverage. This makes it possible to distinguish a 100k source dataset, a small sampled context and a completed 100k model run.

## Results

Results will be added after the CPU runs complete. No version-quality or 100k feasibility claim is made in advance.
