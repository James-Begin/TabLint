# Pre-registrations

These protocols were written before the runs they govern. The files remain unchanged; SHA-256 prefixes identify their frozen versions. Errata are recorded separately in [known issues](KNOWN_ISSUES.md).

| Protocol | SHA-256 (first 16) | Governs | Results |
| --- | --- | --- | --- |
| [Numerical and label confirmation](PROOFREAD_PREREG.md) | `68d76e6ab0da2d86` | H6 (cells), H7 (labels) | [Numerical results](PROOFREAD_RESULTS.md) |
| [Baseline and threshold addendum](PROOFREAD_ADDENDUM_PREREG.md) | `60c7ed47a4253c9c` | Unsupervised baseline; default threshold; pre-run dtype amendment | [Comparison](COMPARISON_TABPFN_EXTENSIONS.md) |
| [Model-version comparison](VERSION_COMPARISON_PREREG.md) | `3f5202dfb4b72004` | H9 (3.5 vs v2), H10 (3.5 vs v3), Fast vs 3.5 | [Model comparison](VERSION_COMPARISON.md) |
| [Familiar-dataset scan](FAMOUS_DATASETS_PREREG.md) | `1b63114979cebfb2` | Exploratory scan protocol and Iris known-answer check | [Gallery](FAMOUS_DATASETS.md) |
| [Categorical confirmation](CATEGORICAL_PREREG.md) | `15aa4b235d76fd07` | H11 (categorical cells) | [Categorical results](CATEGORICAL_RESULTS.md) |

Historical pilot paths mentioned inside a frozen protocol refer to the original project layout. Those pilots are excluded from the confirmation results and are available in Git history. Current reproduction scripts live in [benchmarks/](../benchmarks/README.md).

The numerical showcase’s 12-dataset subset was selected after analysis; it is not a pre-registered subset. [Its selection and statistics](BENCHMARKS.md) are documented separately from the complete confirmation suite.
