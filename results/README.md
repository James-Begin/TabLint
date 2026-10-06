# Stored evidence and reports

These artifacts let reviewers inspect the measurements behind the submission and reproduce figures without GPU inference. Retained run files and current protocol copies support reproduction; they are not new inference on a user’s data.

| Directory | Contents |
| --- | --- |
| [proofread/](proofread/) | 60 numerical/label runs (12 datasets × 5 seeds), summary and analysis |
| [categorical/](categorical/) | 50 categorical confirmation runs (10 datasets × 5 seeds), protocol and summary |
| [proofread_addendum/](proofread_addendum/) | 60 baseline/threshold runs and their analysis |
| [version_compare/](version_compare/) | 60 model-version comparison runs and summary |
| [retail_cpu/](retail_cpu/) | Local CPU capability pilot: descriptions, identifiers, scaling and resource-limit outcomes; distinct from current product integration |
| [high_cardinality/](high_cardinality/) | 15 supplied CPU tables (three datasets × five seeds), 60 v3/3.5 context-arm records, checksum-verified originals and independently computed summary/CSV |

[Graphs and interpretation](../docs/BENCHMARKS.md) · [Scope amendment](../docs/BENCHMARK_AMENDMENT.md) · [Reproduction commands](../benchmarks/README.md)

All 12 retained numerical datasets are included. Two dataset executions were removed after reported test errors; [the amendment](../docs/BENCHMARK_AMENDMENT.md) documents the correction. Retained raw measurements are unchanged. Raw benchmark tables are downloaded by the runners; this directory stores measured outputs, not TabPFN weights.

The [high-cardinality study](../docs/HIGH_CARDINALITY_RESULTS.md) is a separate experiment under TabPFN 9.0.0 defaults, not a measurement of the current product's category-cap override. Its raw JSONs are copied unchanged from the uploaded archive, with [SHA-256 provenance](../benchmarks/high_cardinality_protocol/provenance.json). The analysis includes every specified seed and arm; pilot seed 999 is outside the supplied protocol.
