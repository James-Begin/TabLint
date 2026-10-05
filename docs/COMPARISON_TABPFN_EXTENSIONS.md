# TabLint vs tabpfn-extensions

`tabpfn-extensions` (Prior Labs) is a collection of add-ons for TabPFN: interpretability (SHAP), many-class, embeddings,
p-value tests (CRT), Bayesian optimisation, survival, hurdle regression, conformal prediction with missing data,
TabEBM augmentation, images, and an **unsupervised** module (outlier detection, imputation, synthetic data).
Only the unsupervised module overlaps with TabLint.

## Feature comparison

| | tabpfn-extensions `TabPFNUnsupervisedModel` | TabLint |
|---|---|---|
| Question answered | "How unusual is this row?" (joint density) | "Which cell in this row is wrong, what should it be, and why?" |
| Unit flagged | Row | Cell (and label) |
| Method | Chain-rule density over random feature orderings (default 10), each conditional from TabPFN | Each column predicted from all other columns (+ label) by TabPFN; two-sided tail probability of the recorded value |
| Self-inclusion | Fitted and scored on the same table (documented usage), so a row is part of its own context | Out-of-fold (5 folds), so a row never sees itself |
| Suggested correction | No (`impute` fills cells that are already NaN) | Yes: TabPFN median and 80% interval for every flagged cell |
| Likely-cause hints | No | ×10 / ÷10 / ×1000 slips, swapped digits, sign flip, 0-for-missing, placeholder codes |
| Label errors | No | Yes (out-of-fold probability of the recorded label) |
| Output | Tensor of log-densities | Ranked issue table, Markdown report, highlighted HTML, JSON, app, MCP tools |
| GPU in tabpfn 9.1.0 | `outliers` crashes on CUDA (`docs/KNOWN_ISSUES.md`) | Works (CPU and CUDA) |

## Measured head-to-head (pre-registered addendum; 14 datasets × 5 seeds; `docs/PROOFREAD_RESULTS.md`)

| | TabLint | `outliers` (10 permutations) |
|---|---|---|
| Find rows that contain an injected error (row AUROC, its home turf) | **better on 14/14 datasets**, mean +0.26, p = 0.00006 | |
| Find the erroneous cells (precision@k) | 0.34–0.87 | 0.03–0.25 |
| Median runtime per 400-row table (GPU) | 20 s | 45 s |

Caveats: the extension needed a float64 shim to run on GPU, and it was used as documented (in-sample). Its row AUROC
falls below 0.5 on three tables, consistent with corrupted rows partly explaining themselves. It is a general anomaly
detector, not built for localising data-entry errors, so this compares fitness for TabLint's task, not overall quality.

## Complementary, not competing

- TabLint could be contributed as a cell-level mode of the unsupervised extension, or use its density for row triage.
- Its `impute` could serve as an alternative correction suggester. We have not evaluated that.
- Both packages build on the same capability: TabPFN-3.5's calibrated conditional distributions with no training step.
