# Benchmark execution-error amendment

On 2026-10-05, the project author reported test-execution errors affecting two numerical benchmark datasets and requested removal of their results. The exact failure mechanism was not supplied, so this repository does not attribute their scores to model performance.

The exclusion applies consistently to numerical cells, labels, the optional unsupervised-baseline/threshold analysis and the model-version comparison. Each published numerical suite now covers **12 datasets × 5 seeds = 60 tables**. The categorical experiment is unchanged: **10 datasets × 5 seeds = 50 tables**.

Affected raw run files have been removed from the current tree. Every summary, confidence interval, significance test, threshold statistic, runtime aggregate, table and figure is recalculated from the retained runs. This is an analysis correction using existing measurements, not fresh model inference. Retained raw runs are unchanged.

The hypotheses, injection procedure and scoring rules were specified before the original runs. The retained dataset scope was amended after analysis, and should not be described as a new, prospectively registered 12-dataset experiment. Bootstrap intervals and p-values are conditional on the retained scope and do not model uncertainty about the execution-error exclusions. Independent replication remains useful.

The historical protocol documents were removed from the current tree during repository cleanup. Their original versions remain identifiable in Git history by these SHA-256 prefixes:

| Original protocol | SHA-256 prefix |
| --- | --- |
| Numerical and label confirmation | `68d76e6ab0da2d86` |
| Baseline and threshold addendum | `60c7ed47a4253c9c` |
| Model-version comparison | `3f5202dfb4b72004` |

The current scripts and this amendment document the retained scope. The categorical and familiar-dataset experiments are unchanged.

[Current benchmark results](BENCHMARKS.md) · [Numerical and label results](PROOFREAD_RESULTS.md) · [Version comparison](VERSION_COMPARISON.md)
