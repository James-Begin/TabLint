# Retail benchmark handoff

## Question

Does local TabPFN-3.5 improve price checking and corrections over V2/V3 when a table includes product descriptions, high-cardinality product/customer identifiers and more than 100,000 training rows?

This compares model checkpoints directly, **not the current TabLint UI**. TabLint currently excludes descriptions and high-cardinality IDs from model context; positive results would justify a follow-up integration. V3 already supports large tables and text-related tasks, so these inputs are not exclusive to 3.5. The experiment should measure an improvement, not assume one.

## Data and split

Use [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), Daqing Chen, CC BY 4.0 ([DOI](https://doi.org/10.24432/C5CG6D)).

- Original: **1,067,371 transactions**; cleaned: **779,425**.
- **4,631 stock codes**, **5,878 customer IDs**, **5,241 distinct descriptions**.
- Descriptions are short product names, median **27 characters**; this is not a long-document text benchmark.
- Remove cancelled invoices, nonpositive quantity/price, missing descriptions/customers and exact duplicates.
- Train on dates **before 2011-08-01**; hold out dates **on or after 2011-08-01**. Remove any invoice spanning the boundary.
- Training pool: **583,978** rows; future test pool: **195,447** rows.
- Use nested random training contexts of **1,000 / 5,000 / 10,000 / 100,001 rows**. Keep **1,024 future test rows** fixed within each seed.
- Seeds: **1101, 1102, 1103**. Each model receives the same rows and errors for a given seed and context size.

The source checksum, exact settings and cleaning/split profile are in [`retail_config.json`](../benchmarks/retail_config.json) and [`dataset_profile.json`](../results/retail_cpu/dataset_profile.json). Do not silently reduce the 100,001-row context to fit memory.

## Prediction and error detection

Predict **log(UnitPrice)** using:

| Input | Representation |
| --- | --- |
| Description | pandas string → current-package character TF-IDF / SVD text encoding |
| StockCode | Explicit categorical identifier |
| CustomerID | Explicit categorical identifier |
| Country | Categorical |
| Quantity | Numeric |
| InvoiceDate | Datetime → current-package date encoding |

Learn categorical levels and text/date preprocessing from training rows only. Unseen test IDs become missing values. Record the fraction of test SKUs present in the sampled training context. Exclude invoice number and unit price from features; invoice number is used only to prevent split leakage.

Training prices remain unchanged. Inject synthetic errors into **31 of the 1,024 test prices** (rounded 3%), cycling through:

1. Multiply price by **10**.
2. Divide price by **10**.
3. Swap with another test price that differs by at least **2×**.

Original observed prices are references for these artificial errors, not independently audited ground truth. The corruption pattern stays fixed across models, context sizes and ablations for each seed.

For each recorded test price, use the predicted distribution's CDF to compute:

```text
F = P(log true price <= log recorded price | row context)
surprise = -log10(max(2 × min(F, 1 − F), 1e-9))
suggested price = exp(predicted median log price)
```

Rank by surprise. The 10th–90th percentile interval measures coverage on uncorrupted test values. Scoring uses the model's CDF with matching tensor dtypes; it does not call the optional extension baseline that exhibited the CUDA dtype bug.

## Models, controls and metrics

Compare local **V2 / V3 / V3.5 Base / V3.5 Fast** with `tabpfn==9.1.0`, **one estimator**, identical seeds, shared text/date preprocessing, and **no internal row subsampling**. This isolates checkpoint differences within the current package, not the historical V2 software stack. Remaining version-specific defaults are retained. It does not test API-only Plus or Thinking models.

Include the **SKU median/MAD baseline**, trained on the same rows. For an unseen SKU it uses global statistics; its anomaly score is the absolute log-price deviation divided by a robust scale. This is a strong comparator for repeated retail prices.

Primary inspection:

- **Precision@31:** how many of the top 31 flags are injected errors?
- **AUROC:** how well are injected and unmodified prices separated?
- **Clean log MAE:** how accurately are ordinary held-out prices predicted?
- **Correction log MAE:** how close are suggested replacements to the original prices on injected-error rows?

Also retain price-unit MAE, recall by corruption type, 80% interval coverage, fit/predict/worker time, peak host RSS and CUDA memory where available. Synthetic ground truth measures injected-error detection; an unusual original price is not necessarily a false claim by the model.

At matched context sizes, repeat the V3/V3.5 variants with **Description removed** and with **StockCode + CustomerID removed**. Keep all rows, targets, errors, seeds and other settings identical. These ablations show whether descriptions or identifiers actually help.

Report all completed versions and all failed attempts. Use matching completed seeds for paired comparisons; do not treat an unsupported or timed-out model as having zero accuracy. Three seeds provide an exploratory comparison, not a strong significance claim. Do not select only the metric, subset or seed where 3.5 wins.

## Run on another machine

From a fresh clone, or pull the latest main:

```bash
git clone https://github.com/James-Begin/TabLint.git
cd TabLint
uv sync --locked
uv run --with openpyxl python -m benchmarks.retail_cpu --prepare
uv run python -m benchmarks.retail_cpu --download
```

Model downloads require Prior Labs access/license acceptance on that machine. Inference uses local weights, not an API. The runner explicitly selects CPU by default; `--device cuda:0` selects a CUDA GPU.

Example GPU run below assumes **at least 32 GiB host RAM** so a 24 GiB host-process cap leaves headroom. Adjust the host cap to the actual machine; this is not a GPU VRAM limit. The CUDA path is prepared but has not been validated on this device.

```bash
# Scaling and matched checkpoint comparison.
uv run python -m benchmarks.retail_cpu \
  --device cuda:0 --profile standard \
  --sizes 1000,5000,10000,100001 \
  --seeds 1101,1102,1103 \
  --out results/retail_gpu \
  --timeout 3600 --rss-limit-gib 24

# Text / identifier ablations at the largest context.
# V2's supported sample limit excludes this size.
uv run python -m benchmarks.retail_cpu \
  --device cuda:0 --profile standard \
  --sizes 100001 --seeds 1101,1102,1103 \
  --versions V3,V3_5,V3_5_FAST \
  --ablations no_text,no_ids \
  --out results/retail_gpu \
  --timeout 3600 --rss-limit-gib 24

# Rebuild the report, CSV, JSON summary and graphs from the raw outputs.
uv run --with matplotlib==3.11.2 python -m benchmarks.summarize_retail \
  --root results/retail_gpu \
  --report docs/RETAIL_GPU_RESULTS.md \
  --figure docs/figures/retail_gpu
```

For a larger CPU machine, replace `--device cuda:0` with `--device cpu`, use a fresh `--out results/retail_cpu_large`, and set the RAM/time budgets appropriately. The filename `retail_cpu.py` reflects its initial CPU pilot; the runner now accepts either device. Use a fresh output directory for a new machine, precision, profile or time budget. Existing records are skipped, including failures; use another directory to retry.

V2 is deliberately not forced beyond its supported sample limit. GPU OOMs, host memory-limit failures and timeouts are recorded. Check `status`, `train_rows`, `model_subsample_samples`, `expanded_feature_count` and `n_estimators_actual` before claiming a completed 100k comparison. Hardware and package versions are written to `environment.json`; preserve it with the results.

## CPU pilot so far

On the local Apple M5, 10 CPU cores / 16 GiB RAM, all four checkpoints completed the **1k, 5k and 10k** contexts for seed 1101. At 10k, V3.5 Base's clean log MAE was **0.205 vs V3's 0.272** (about 25% lower), and precision@31 was **0.613 vs 0.581**. The SKU baseline's precision@31 was **0.710**, so the pilot does not establish universal superiority over alternatives.

The standard 100,001-row attempts exceeded the 8 GiB host RSS budget for V3 and both 3.5 variants; V2 rejected the input above its 10k supported limit. A separately frozen [compact text retry](../benchmarks/retail_compact_cpu.json) reduces description SVD dimensions to eight. V3 and both 3.5 variants also exceeded the budget in that retry; no 100k TabPFN quality result was completed locally. Do not pool this profile with standard results.

[CPU report and graph](RETAIL_CPU_BENCHMARK.md) · [Raw CPU measurements](../results/retail_cpu/)
