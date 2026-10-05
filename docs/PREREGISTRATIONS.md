# Pre-registrations

Each file was written before the run it governs; SHA-256 prefixes identify the frozen versions.
Paths inside them refer to the original layout: `pivot/` is now `benchmarks/` and `results/pivot/` is now `results/pilots/` (renamed after the runs; contents unchanged).

| File | SHA-256 (first 16) | Governs |
|---|---|---|
| `docs/REGIME_SCREEN_PREREG.md` | 15ff3f55c031bdb0 | Regime screen (where TabPFN beats logistic regression) |
| `docs/CONFIRMATION_PREREG.md` | 95d5e4472f70082e | Stress test H1 |
| `docs/CONFIRMATION2_PREREG.md` | 072d4b74fffaffec | Stress test H2 |
| `docs/COMPARISONS_PREREG.md` | 0e6360f78aed6e45 | Stress test model comparisons (descriptive) |
| `docs/REGIME_CONFIRMATION_PREREG.md` | f47894a9d6815d82 | Stress test H4, H5 |
| `docs/PROOFREAD_PREREG.md` | 68d76e6ab0da2d86 | Proofread H6 (cells), H7 (labels) |
| `docs/PROOFREAD_ADDENDUM_PREREG.md` | 60c7ed47a4253c9c | Proofread vs Prior Labs outlier detector; default threshold (amended before results: dtype shim) |

**Errata.** Corrections to the wording of the PROOFREAD_ADDENDUM amendment (CUDA-only crash; float64 precision) are recorded in `docs/KNOWN_ISSUES.md`.
