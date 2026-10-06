# Stored evidence and reports

These artifacts let reviewers inspect the measurements behind the submission and reproduce figures without GPU inference. Retained run files and current protocol copies support reproduction; they are not new inference on a user’s data.

| Directory | Contents |
| --- | --- |
| [proofread/](proofread/) | 60 numerical/label runs (12 datasets × 5 seeds), summary and analysis |
| [categorical/](categorical/) | 50 categorical confirmation runs (10 datasets × 5 seeds), protocol and summary |
| [proofread_addendum/](proofread_addendum/) | 60 baseline/threshold runs and their analysis |
| [version_compare/](version_compare/) | 60 model-version comparison runs and summary |
| [famous/](famous/) | Exploratory real-table reports and source-check notes, including negative and unverified findings |
| [proofread_demos/](proofread_demos/) | Saved reports used by UI tests and production rehearsals; not additional benchmark evidence |

[Graphs and interpretation](../docs/BENCHMARKS.md) · [Scope amendment](../docs/BENCHMARK_AMENDMENT.md) · [Reproduction commands](../benchmarks/README.md)

All 12 retained numerical datasets are included. Two dataset executions were removed after reported test errors; [the amendment](../docs/BENCHMARK_AMENDMENT.md) documents the correction. Retained raw measurements are unchanged. Raw benchmark tables are downloaded by the runners; this directory stores measured outputs, not TabPFN weights.
