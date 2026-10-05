# Categorical-cell checks — pre-registration (H11)

Written: 2026-10-04 13:21 EDT, before the benchmark below. One development smoke test on Auto MPG with 3 planted
errors was run during implementation (all 3 ranked 1, 2, 4). Auto MPG is therefore **excluded** from this test.

## Method under test (shipped code)
`Proofreader.categorical_columns` + `Proofreader.categorical_scores` (proofread/core.py, at the hashes recorded in
results/categorical/protocol.json at launch). Categorical columns: text/bool/category columns with 2–30 levels (at most
half as many levels as rows), plus numeric columns with 2–10 distinct values. Score = −log10 out-of-fold
TabPFN-3.5 P(recorded category | rest of row), 5 folds, n_estimators 4.

## Datasets (fixed): 10 public tables
OpenML: credit-g (31), adult (1590), mushroom (24), car (40975), bank-marketing (1461), nursery (26), cmc (23),
kr-vs-kp (3); seaborn-data: penguins, tips. No label column (every categorical column is checked).

## Design
Seeds 901–905. Per seed: 400 random rows; categorical columns found by the shipped rule on the clean sample; 3% of
categorical cells corrupted by a **swap**: replaced with a different level of the same column, drawn from that column's
empirical distribution (a plausible value for the column, not a typo). Baselines (all out-of-fold, 5 folds, features =
all other columns, categoricals one-hot or native): **frequency** (−log of the recorded level's frequency in its
column), **logistic regression**, **random forest**, **kNN** (k=10), **HGB** (native categorical support); score =
−log P(recorded level).

## Hypothesis
H11: precision@k (k = number of injected errors) of TabPFN exceeds that of the per-dataset best of the 5 baselines
(chosen after the fact). Unit = dataset (mean over seeds). One-sided Wilcoxon signed-rank over the 10 datasets,
alpha 0.05. Also reported: wins, mean difference with bootstrap CI, AUROC. Reported whatever the outcome.
