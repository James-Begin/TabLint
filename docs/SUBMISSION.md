# Submission draft (hackathon form, ≤ 5,000 characters)

## Title
TabLint — spellcheck for tables, powered by TabPFN-3.5

## Project description

Every analyst has shipped a model trained on a typo: a BMI of 0 standing in for "missing", a radius of 211 that should
have been 21.1, a value pasted from the wrong row. Per-column checks (z-scores, range rules) only catch values that are
extreme *for the column*. Most real slips are ordinary for the column and impossible *for the row*.

**TabLint** reads a table the way a careful analyst would. For every cell, TabPFN-3.5 predicts the value from the
rest of its row, in context and out of fold, with no training or tuning. Because TabPFN returns a **full predictive
distribution**, TabLint knows how surprising each value is *for that row*. It underlines suspicious cells and labels,
suggests the likely value, and names the probable slip when one fits: ×10 or ÷10 decimal, swapped digits, sign flip,
or 0 for missing. Each flag lists the related columns in the row, so a human can verify it in seconds. Column-level
placeholder codes (e.g. 0 in 49% of insulin values) are reported once instead of as hundreds of flags.

**Evidence (pre-registered, every dataset reported).** We wrote the hypotheses, datasets, error types, baselines and
test before running the confirmation: 14 public tables × 5 fresh seeds, 3% of cells corrupted with five realistic error
types, 10% of labels flipped. We compared against the best baseline for each dataset, chosen after the fact (a
deliberately hard bar).
- Cell errors: TabPFN-3.5 beats the best of 7 baselines (z-score, IsolationForest, ridge, kNN, random forest, gradient
  boosting, and quantile gradient boosting, which also models per-row uncertainty) on 13/14 datasets. Mean precision@k
  is +0.19 [+0.13, +0.24], Wilcoxon p = 0.0002. On 13 of the 14 datasets, 59–87% of its top flags are real errors.
- It is best on all five error types. The gain is largest on swapped values that look normal for their column: AUROC
  0.85 vs 0.79 for the best baseline, and 0.56 for z-score.
- The predictive distribution matters: it beats TabPFN's own point-prediction residual on 14/14 datasets.
- Suggested fixes are closer to the truth than random-forest predictions on 13/14 datasets.
- Against Prior Labs' own TabPFN outlier detector (tabpfn-extensions), which scores whole rows: TabLint is better at
  finding rows with errors on 14/14 datasets (AUROC +0.26, p = 0.00006), and it also pinpoints the cell.
- At the default threshold, 88% of listed cells are real errors (53% recall). That figure is from a pre-registered
  addendum.
- Label errors: TabPFN-3.5 beats the best of 4 baselines on 13/14 (mean AUROC +0.022, p = 0.0009), with large gains
  on nonlinear data (+0.13 on eeg-eye-state).
- Where it doesn't help: a table of independently sampled simulation parameters. With no structure between columns,
  a z-score wins. We report it.
- Real data: on the raw Pima diabetes table, the top flags include the known impossible zeros (BMI, glucose, blood
  pressure) and a 99 mm skin fold.

**What ships.** A Jupyter-first workflow: install the notebook extra, then `df.tablint.view()` checks your own DataFrame and returns reviewed results to Python. A terminal spreadsheet viewer (`proofread view data.csv`): flagged cells are highlighted, you accept or dismiss each fix, and it saves a cleaned CSV plus an audit log. A Jupyter widget and pandas accessor: `df.tablint.view()` lets you click, accept and dismiss fixes, and returns the cleaned DataFrame to Python. A VS Code extension shows squiggles on bad CSV cells with one-click fixes (integration-tested in real VS Code). Also a Python library (`Proofreader().check(df)`), a CLI (`proofread check data.csv` writes a readable
report, a highlighted HTML table and a CSV of issues), an interactive app that shows TabPFN's plausible range against
the recorded value with live ground-truth scoring on demo tables, and an MCP server so an LLM data-cleaning agent can
review a table, inspect each flagged row, and write a cleaning memo grounded in TabPFN's numbers. A 400 × 30 table
takes 1–2 minutes on one GPU.

**Rigor.** Pre-registration, all runs stored as JSON, generated results tables, automated unit and integration checks, clean-copy reproduction. We
also report directions that failed: gradient-based label scores, full conformal prediction, and look-ahead active
learning.

**Also included:** a research module that differentiates through TabPFN-3.5's in-context learning to stress-test single
decisions. In pre-registered tests, gradient search beat matched black-box search on 3 of 7 datasets (e.g. 93% vs 43%
verified flips). Along the way we found and fixed NaN gradients in TabPFN's preprocessing.

Code: Apache-2.0. Data: public UCI/OpenML tables, loaded by ID.

## Video script

The full shot list, with exact commands and verified on-screen numbers, is in `docs/DEMO_SCRIPT.md` (Auto MPG demo).


## Submission assets

- [Approved 1:28 product showcase](../demo/showcase/TabLint-demo.mp4)
- [Quick start](GETTING_STARTED.md): installation in Jupyter, analysis of your own data and export.
- [Technical summary and features](../README.md)
- [Video provenance](../demo/showcase/README.md): measured numerical report, illustrative categorical suggestion.
