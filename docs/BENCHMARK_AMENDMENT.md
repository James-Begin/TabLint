# Benchmark execution-error amendment

On 2026-10-05, the project author reported test-execution errors affecting two numerical benchmark datasets and requested removal of their results. The exact failure mechanism was not supplied, so this repository does not attribute their scores to model performance.

The exclusion applies consistently to numerical cells, labels, the optional unsupervised-baseline/threshold analysis and the model-version comparison. Each published numerical suite now covers **12 datasets × 5 seeds = 60 tables**. The categorical experiment is unchanged: **10 datasets × 5 seeds = 50 tables**.

Affected raw run files have been removed from the current tree. Every summary, confidence interval, significance test, threshold statistic, runtime aggregate, table and figure is recalculated from the retained runs. This is an analysis correction using existing measurements, not fresh model inference. Retained raw runs are unchanged.

The hypotheses, injection procedure and scoring rules were specified before the original runs. The retained dataset scope was amended after analysis, and should not be described as a new, prospectively registered 12-dataset experiment. Bootstrap intervals and p-values are conditional on the retained scope and do not model uncertainty about the execution-error exclusions. Independent replication remains useful.

The current scripts, stored run metadata and methodology describe the retained scope. Historical protocol drafts are not distributed in this submission snapshot. The categorical experiment is unchanged. The repository has been consolidated to one commit for submission; this snapshot does not offer a Git-history record of pre-run registration.

[Current benchmark results](BENCHMARKS.md) · [Numerical and label results](PROOFREAD_RESULTS.md) · [Version comparison](VERSION_COMPARISON.md)
