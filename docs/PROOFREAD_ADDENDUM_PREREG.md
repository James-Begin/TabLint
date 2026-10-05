# Proofread addendum — pre-registration

Written: 2026-10-03 18:18 EDT, before running. Same 70 corrupted tables as docs/PROOFREAD_PREREG.md
(regenerated deterministically from seeds 701–705; Proofread's stored precision@k must reproduce, which checks determinism).

## A1. Prior Labs' TabPFN outlier detector as a baseline
tabpfn-extensions 0.6.3 `TabPFNUnsupervisedModel.outliers` (default 10 feature permutations; TabPFN-3.5 classifier and
regressor; fitted on the corrupted table; score = −log density per row). It is a row-level detector, so:
- Cell level: row score broadcast to every cell of the row; precision@k and AUROC as in H6.
- Row level (its home turf): AUROC for "row contains ≥1 injected error". Proofread's row score = max cell surprise.
Reported per dataset with the number of datasets where Proofread is higher; one-sided Wilcoxon over 14 datasets for the
row-level comparison (A1-row), alpha 0.05. No other analyses decided in advance.

## A2. Operating point actually shown to users
Proofread's default lists cells with surprise ≥ 3 (−log10 two-sided tail probability; i.e. tail prob ≤ 0.001).
Report precision (share of listed cells that are real errors) and recall (share of injected errors listed) at thresholds
3 and 2, per dataset and pooled. Descriptive; no test.

## Amendment (before any result): compatibility shim
tabpfn-extensions 0.6.3 fails with tabpfn 9.1.0 (float32 targets vs float64 bar-distribution borders in criterion.forward).
The baseline regressor's logits are upcast to float64 before the extension uses them; no values are changed.
