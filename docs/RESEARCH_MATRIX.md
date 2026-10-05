# Experimental matrix and scope

## Lock-in priorities
1. A validated, replayable counterexample report on standard TabPFN (not merely surrogate).
2. Matched query-budget and wall-time baselines: random constrained search, coordinate search, gradient search.
3. Independent context seeds, with clean accuracy and failed/invalid-run denominators.
4. Numeric and categorical schemas declared, immutable features explicit, poisoned labels clearly shown.
5. Repair/defense comparison: exact duplicate aggregation; regularized trees; verification after mitigation.

## Model-under-the-hood ablations
- **Fingerprint**: default vs disabled (record all settings). Disentangle changed-table hashing effects from row influence.
- **Feature scaling**: raw vs context-standardized differentiable path. Evaluate finite-gradient fraction and deployed transfer, not just surrogate score.
- **Soft-label interpolation**: integer equivalence + finite-difference tests; suitable for label-sensitivity visualization, not causal attribution.
- **Ensemble**: single estimator vs a small gradient ensemble; verify at receiver n_estimators=4. Record compute cost.
- **Discrete projection**: valid one-hot plus immutable categories; relaxed proposals hard-verified, never count relaxed success.
- **Gradient handling**: abort default; masking is an exploratory workaround. Count masked dimensions and record explicit diagnostics.
- **Fine-tuning / robust adaptation**: optional, only after stable verification engine. Context/task-wise split, clean performance constraint, unseen attack evaluation. Compare simple duplicate handling before expensive fine-tuning. Fine-tuning is not justified just because model modification is possible.

## Datasets
- German credit: mixed categories, UCI CC BY 4.0; pilot categorical failure retained.
- Taiwan credit default: UCI CC BY 4.0; sex/education/marriage/payment-status codes are categorical semantics even if stored numerically. Never call arbitrary numeric interpolation of those codes schema-valid.
- Wisconsin breast cancer diagnostic: sklearn bundled/UCI CC BY 4.0, numeric; intended reproducible nonclinical demonstration.
- Pima diabetes: exploratory pilot only until redistribution rights resolved. Historical zero-valued measurements are not assured valid physiology.

## Honest narrative
The showcase is differentiation THROUGH learning and exact verification, not a first-ever data-poisoning attack or a claim that trees cannot be audited. Synthetic labels are adversarially corrupted. No found counterexample is a robustness certificate. Closest-real-row bounds do not assure real-world plausibility. Transfer to a receiver is evaluated, not assumed.

## Demo vs benchmark
A demo may display a successful example chosen from development cases, but must label it as curated and link all-run rates. Never present curated success as expected performance. Replay artifacts include failed examples too and are available without a GPU or model download.
