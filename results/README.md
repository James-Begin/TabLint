# Stored evidence and reports

These artifacts let reviewers inspect the measurements behind the submission and reproduce figures without GPU inference. Original run files and frozen protocols are preserved; they are not new inference on a user’s data.

| Directory | Contents |
| --- | --- |
| [proofread/](proofread/) | 70 numerical/label confirmation runs (14 datasets × 5 seeds), summary and analysis |
| [categorical/](categorical/) | 50 categorical confirmation runs (10 datasets × 5 seeds), protocol and summary |
| [proofread_addendum/](proofread_addendum/) | 70 baseline/threshold runs and their analysis |
| [version_compare/](version_compare/) | 70 model-version comparison runs and summary |
| [famous/](famous/) | Exploratory real-table reports and source-check notes, including negative and unverified findings |
| [proofread_demos/](proofread_demos/) | Saved reports used by UI tests and production rehearsals; not additional benchmark evidence |

[Graphs and interpretation](../docs/BENCHMARKS.md) · [Protocols](../docs/PREREGISTRATIONS.md) · [Reproduction commands](../benchmarks/README.md)

The README’s numerical view covers a post-analysis selection of 12 datasets. All 14 remain here for transparency. Raw benchmark tables are downloaded by the runners; this directory stores measured outputs, not TabPFN weights.
