# Proofread: cell- and label-error detection with TabPFN-3.5 — pre-registration

Written: 2026-10-03 16:36 EDT, before any confirmation run. Pilots (results/pivot/cells,
results/pivot/label_pilot; seeds 101–103) motivated this and are excluded.

## Datasets (fixed): 14 public numeric tables
eeg-eye-state, MagicTelescope, breast-cancer, phoneme, wilt, electricity, diabetes, banknote, spambase,
climate-crashes, kc1, steel-plates, ilpd, blood-transfusion. Continuous columns only (>10 distinct values).
Per seed: 400 random rows. Seeds 701–705.

## Experiment C (cells)
Inject errors into 3% of cells, types cycled: decimal (×10 or ÷10), swap (value from another row, same
column, ≥0.5 SD different), offset (±3 SD), zero (value set to 0, only if ≥0.5 SD from 0, else offset),
transposition (swap the two leading significant digits, only if the change is ≥0.5 SD, else decimal).
All per-cell scores are 5-fold out-of-fold, each column predicted from the other columns + label:
zscore (median/MAD), iforest (row score broadcast), ridge, knn (k=10), rf, hgb (standardized |residual|),
hgb_quantile (|x − q50| / (q90 − q10)), and tabpfn_pit (two-sided tail surprise of the observed value under
TabPFN-3.5's predictive distribution). Also tabpfn_resid (ablation).
H6 (primary): precision@k (k = number of injected errors) of tabpfn_pit exceeds that of the BEST baseline
per dataset (best chosen post hoc among the 7 non-TabPFN scores — conservative). Unit = dataset (mean over
seeds). Test: one-sided Wilcoxon signed-rank over the 14 datasets, alpha 0.05. Also reported: number of
datasets won, bootstrap CI of the mean difference, AUROC, per-error-type AUROC.
Secondary: correction error (|TabPFN median − true| in column SDs) vs column median and vs RF / ridge
predictions.

## Experiment L (labels)
Same datasets/seeds, 400-row tables, flip 10% of labels uniformly at random. Scores: 1 − P(given label)
out-of-fold (5 folds) for logistic, kNN (k=10), RF, HGB, and TabPFN-3.5.
H7: AUROC of TabPFN exceeds the per-dataset best of the 4 baselines (post hoc best; conservative).
Same test as H6. Both H6 and H7 are reported regardless of outcome; no further datasets will be added.
