# TabLint benchmark showcase plan

## Lead with the complete numerical result

**Proposed headline:** “Find more real errors in the cells you review.”

**Evidence:** TabLint with TabPFN-3.5 beats the strongest tested baseline for each dataset on **13 of 14 numerical datasets**, with a **19.0 percentage-point mean gain in precision@k**. The 95% bootstrap interval for that mean gain is **13.2–24.0 points**. This is the primary, pre-registered result: 14 public datasets, five seeds each, 70 tables, 400 rows per table, 3% injected cell errors.

Precision@k means the share of the top k flagged cells that are injected errors, where k is the number of injected errors. It measures the quality of the ranked review list. It is not ordinary classification accuracy, the precision of the default product threshold, or a guaranteed error-detection rate on a user's data.

Use the full benchmark as the main claim, then use a selected dataset to make the numbers tangible.

## Featured numerical example: breast-cancer

**Select breast-cancer for the primary detailed comparison.** Its five-seed mean precision@k is **84.7% for TabLint versus 57.0% for histogram gradient boosting**, the strongest of the seven tested alternatives on this dataset: **+27.7 percentage points**.

This example combines a large detection improvement with high absolute precision and better correction estimates. The mean absolute correction error is **0.155 column standard deviations for TabPFN**, versus **0.426 for random forest**, **0.543 for ridge**, and **0.744 for the column median**. Correction error is measured on injected errors against their original clean values; lower is better. This does not measure exact-match correction success.

Selection is editorial and made after reviewing results. It is not a separately pre-registered winning example. Blood-transfusion has the largest detection gain (+31.3 points, 78.2% versus 46.9%); kc1 has the highest numerical precision (87.0% versus 63.1%, +23.9 points). Breast-cancer offers a useful balance of a large gain, high precision, and a clear correction comparison. These are tests of corrupted table cells, not diagnostic or clinical performance.

**Suggested caption:** “On the breast-cancer table, 84.7% of the top-ranked cells were injected errors, compared with 57.0% for the strongest tested alternative. Means over five independently corrupted 400-row samples.”

### Values for the numerical spotlight graph

| Method | Precision@k |
| --- | ---: |
| TabLint / TabPFN-3.5 predictive distribution | 84.7% |
| Histogram gradient boosting | 57.0% |
| k-nearest neighbours | 46.2% |
| Ridge | 41.9% |
| Random forest | 40.5% |
| Per-column robust z-score | 38.7% |
| Histogram gradient boosting quantiles | 6.1% |
| Isolation Forest row score, broadcast to cells | 4.4% |

Display all tested alternatives in the detailed graph. The Isolation Forest score was adapted from rows to cells; explain that adaptation in its label rather than implying it is a native cell-localisation method. The result compares the implemented benchmark configurations, not every possible tuned version of these models.

## Categorical example: mushroom

**Categorical headline:** “Categorical values need row context too.” TabPFN wins on **10 of 10 categorical datasets**, with a **6.2 percentage-point mean gain** over each dataset's best tested alternative; 95% bootstrap interval **4.2–8.8 points**. This separate pre-registered experiment has 50 tables, five seeds per dataset, up to 400 rows per sample (penguins: 344; tips: 244), and 3% categorical cells replaced with another category from the same column.

**Select mushroom:** precision@k is **77.0% versus 61.0% for k-nearest neighbours**, the strongest tested alternative: **+16.0 percentage points**, the largest categorical gain. This is a selected example; pair it with the complete ten-dataset comparison. Category-frequency scoring reaches only **7.6%** on this dataset, but it should not be the main comparison because stronger alternatives exist.

| Method | Precision@k on mushroom |
| --- | ---: |
| TabLint / TabPFN-3.5 | 77.0% |
| k-nearest neighbours | 61.0% |
| Random forest | 60.1% |
| Logistic regression | 57.9% |
| Histogram gradient boosting | 54.2% |
| Category frequency | 7.6% |

**Suggested caption:** “Valid categories can still be wrong for a row. On mushroom, TabLint achieves 77.0% precision among the top-ranked cells, versus 61.0% for the strongest tested alternative.”

These corruptions are valid categories taken from the same column, not misspellings. A replacement consistent with both the column and the rest of the row may be undetectable. The separate spelling heuristic is not evaluated here. Auto MPG is a development example and is excluded from this confirmation test.

## Graphs to build, in presentation order

| Order | Graph | Visual specification | Purpose and placement |
| --- | --- | --- | --- |
| 1 | Complete numerical comparison | Horizontal paired-dot chart, all 14 datasets; TabLint in green, best baseline in slate. Fixed precision@k axis from 0% to 100%. Show baseline method beside each row, +19.0-point mean gain with its interval, and 13/14 wins. Keep climate-crashes visible in a contrasting colour. | The one main chart for the README Evidence section. Establishes breadth before a selected success. |
| 2 | Numerical spotlight | Horizontal bars for the eight methods above, sorted by precision@k, with direct percentage labels and a 0–100% axis. Highlight 84.7% versus 57.0% and annotate +27.7 points. | Detailed benchmark page and submission presentation. A short video can simplify to the two leading bars and name the comparator. |
| 3 | Categorical comparison | Full ten-dataset paired-dot chart with a 0–100% axis; accompany it with a two-bar mushroom inset, 77.0% versus 61.0%. Put “10/10 datasets” and “+6.2 points on average” above the full chart. | Demonstrates that the categorical feature has its own measured evidence. In a video, show the mushroom bars and retain the all-dataset headline. |
| 4 | Suggested-fix quality | Four horizontal bars for breast-cancer: TabPFN 0.155 SD, random forest 0.426, ridge 0.543, column median 0.744. Start the axis at zero; label “Mean absolute correction error / column SD; lower is better.” | Shows that TabLint also proposes useful replacements. Keep this separate from detection precision. |
| 5, optional | Why use the distribution? | Paired-dot chart across the same 14 numerical datasets: TabPFN tail-surprise score versus its own point-prediction residual score. Calculate dataset means from the raw runs. Existing report headline is about 0.72 versus 0.54, with distribution scoring better on 14/14. | Technical appendix or a brief TabPFN explanation slide. This is an ablation within TabPFN, not an external alternative. |

Build the first four graphs as light-background PNG and SVG exports with large type, direct labels, and consistent green/slate colours. Avoid cropped bar axes, 3D bars, mixed precision/AUROC axes, and treating “+19 points” as “19% better.” Add a compact protocol caption to every export so it remains interpretable outside the README.

The README should have one readable main chart, two short numerical/categorical headline sentences, and links to the complete results. Keep the spotlight, correction and ablation graphs in the benchmark document rather than filling the setup page with charts.

## Video / submission sequence

Use a short evidence segment after showing numerical and categorical review in the product:

1. **Numerical result:** reveal the 57.0% baseline bar, then the 84.7% TabLint bar; show “+27.7 percentage points” and the dataset name. Keep the caption “precision@k; five seeds; injected cell errors” visible.
2. **Breadth:** expand to all 14 dataset pairs and show “13/14 numerical datasets · +19.0 points on average.” Include the losing dataset in the expansion.
3. **Categorical result:** reveal 61.0% versus 77.0% for mushroom, then show “10/10 categorical datasets · +6.2 points on average.”
4. **Optional correction beat:** show the four correction-error bars only if there is enough reading time to explain the different metric.

Use about 4–5 seconds for each simple two-bar view, measured after its text is fully rendered; use at least 6 seconds for a complete dataset chart. Keep the dense charts out of a short video unless there is time to read them. This is a production plan; no changes to the approved animation are included in this update.

## Evidence boundaries to keep in the presentation

- Tests use synthetic errors on sampled public tables. Real-data examples demonstrate the interface, not a measured real-world detection rate.
- Each dataset's strongest baseline is chosen after observing its mean across seeds: seven numerical and five categorical alternatives. Show that choice clearly in the caption.
- Numerical predictions use five-fold out-of-fold scoring, with other columns plus the label available as context under the registered protocol. Categorical tests check each categorical column from the rest of the row, without a designated label column. Do not silently present this as an identical task to every user's uploaded CSV.
- Report the climate-crashes numerical loss: 34.2% versus 42.4% for per-column z-score. Independent columns give little useful context to row-based prediction.
- The numeric library's default threshold of 2 has pooled **87.9% precision and 53.3% recall** on these confirmation tables. This is a different operating point from precision@k. If used, show both precision and recall and identify the default-threshold analysis; do not substitute 87.9% into a precision@k comparison.
- Do not headline “better than every TabPFN version”: the separate version comparison does not support 3.5 outperforming v3. Speed claims must name hardware and timing conditions.
- Do not lead with the enormous gap to the TabPFN extensions row-outlier detector. It has a different task and scoring setup. If discussed in a technical appendix, compare row AUROC on both sides and retain the documented compatibility and in-sample scoring caveats.
- Bootstrap intervals in the headline resample dataset-level differences after averaging seeds. They are not per-dataset error bars. Single-dataset spotlight bars may show the five seed values as points; if adding intervals, state how they were computed and that there are only five runs.

## Sources and implementation handoff

| Artifact | Source |
| --- | --- |
| Numerical headline and complete chart | [summary JSON](../results/proofread/summary.json), `h6_cells` and `cells_table` |
| Numerical spotlight, corrections and distribution ablation | `results/proofread/breast-cancer_701.json` through `_705.json`; `cells.precision_at_k` and `cells.correction_err_sd`. Ablation uses the same keys across all 70 numerical runs. |
| Categorical headline and complete chart | [summary JSON](../results/categorical/summary.json), `h11` and `per_dataset` |
| Mushroom spotlight | `results/categorical/mushroom_901.json` through `_905.json`, `precision_at_k`; verify against `per_dataset.all_p_at_k` |
| Protocol and definitions | [Numerical pre-registration](PROOFREAD_PREREG.md), [categorical pre-registration](CATEGORICAL_PREREG.md) |
| Complete reported results | [Numerical results](PROOFREAD_RESULTS.md), [categorical results](CATEGORICAL_RESULTS.md), [version comparison](VERSION_COMPARISON.md) |

Recompute spotlight means from the five raw runs, without rerunning inference or changing benchmark configurations. Use the full-precision raw numbers for calculations and one decimal place for displayed percentages. For all-dataset charts, the existing numerical summary stores means rounded to three decimals; raw runs are preferred when exact plotted values are required.

The implemented renderer lives in `benchmarks/make_showcase_figures.py`, write `docs/figures/tablint_benchmark_*.png` and `.svg`, and export a CSV of plotted values beside the figures. Verify that all 14/10 datasets and 70/50 runs are present, each selected baseline is truly the strongest on the relevant metric, axis directions match the metric, labels do not overlap, and the SVG/PNG versions agree. Inspect each exported graph at README width and at full presentation resolution before publishing.

## Implemented showcase

The six graphs, exported values and reproduction command are now published in [Benchmark graphs and methodology](BENCHMARKS.md). The categorical spotlight is a separate full-size graph so all alternatives remain legible. The README includes the complete numerical comparison.
