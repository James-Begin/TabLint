# TabPFN version comparison on the Proofread cell benchmark — pre-registration

Written: 2026-10-04 10:41 EDT, before any run.

## Question
Does TabPFN-3.5 specifically matter for Proofread's cell-error detection, compared with earlier TabPFN versions run the
same way? Is TabPFN-3.5-Fast an acceptable speed/quality trade-off?

## Versions (fixed)
TabPFN v2, v3, v3.5, v3.5-Fast (tabpfn 9.1.0 `ModelVersion` V2, V3, V3_5, V3_5_FAST; default regressor settings).
v2.5 and v2.6 are **excluded before running**: their weights need an interactive one-time license acceptance that was
not available on the benchmark machine.

## Design
Identical to H6 in docs/PROOFREAD_PREREG.md (same 14 datasets, 400-row tables, continuous columns, 3% of cells
corrupted with the same five error types and the same injection code, 5-fold out-of-fold, score = two-sided tail
surprise under the regressor's predictive distribution), except: **fresh seeds 801–805** (not used before), and only
the TabPFN score is computed, once per version. Metrics: precision@k (k = number of injected errors), cell AUROC,
wall-clock seconds per table (same GPU type, one process per GPU).

## Hypotheses (unit = dataset, mean over seeds; one-sided Wilcoxon signed-rank over 14 datasets; Bonferroni over 2 → alpha 0.025)
- H9: precision@k of v3.5 > v2.
- H10: precision@k of v3.5 > v3.
Descriptive only: v3.5-Fast vs v3.5 (paired difference in precision@k, and speed ratio).
All versions and datasets are reported whatever the outcome. If H9/H10 are not supported, no version-specific claim
is made.
