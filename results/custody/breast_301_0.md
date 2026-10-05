# Chain of Custody — investigation report

Threat model: k rows are appended to the training context with adversarially chosen labels; feature values are schema-valid and bounded near real rows (geometric checks, not domain realism).

## 1. Suspect (differentiable surrogate; not yet verified)
- Query decision on surrogate: P(class 1) = 0.0000 → class 0
- After appending 3 rows: P = 0.0373 → class 0; flip=False, constraints valid=True
- Appended labels: [1, 1, 1]

| Feature | Query | Row 1 | Row 2 | Row 3 |
|---|---:|---:|---:|---:|
| symmetry error | 0.01374 | 0.02057 | 0.02154 | 0.01054 |
| mean texture | 13.78 | 17.3 | 13.27 | 16.33 |
| worst symmetry | 0.2369 | 0.2688 | 0.2823 | 0.2806 |
| smoothness error | 0.006064 | 0.006809 | 0.004348 | 0.003828 |
| worst texture | 17.48 | 21.1 | 16.93 | 20.2 |
| worst smoothness | 0.1298 | 0.1384 | 0.117 | 0.1264 |
| worst radius | 13.5 | 13.71 | 14.67 | 15.85 |
| texture error | 0.6931 | 0.9505 | 0.6946 | 0.4801 |

## 2. Prove (refit on the exact rows and labels)
| Receiver | P before | P after | Decision flipped | Benign-label control flipped |
|---|---:|---:|---|---|
| tabpfn_standard | 0.0000 | 0.0356 | False | False |
| hgb | 0.0000 | 0.0126 | False | False |
| logistic | 0.0003 | 0.0041 | False | False |

**Verified on deployed TabPFN-3.5 (flip and benign-label control clean): False**

## 3. Catch (exact leave-one-out, context + appended rows)
- Appended rows ranked in the top 3: **3/3**

| Rank | Row | |Δp| | Label | Appended |
|---:|---:|---:|---:|---|
| 1 | 201 | 0.0264 | 1 | True |
| 2 | 200 | 0.0176 | 1 | True |
| 3 | 202 | 0.0100 | 1 | True |
| 4 | 181 | 0.0076 | 0 | False |
| 5 | 100 | 0.0058 | 0 | False |
| 6 | 109 | 0.0055 | 0 | False |
| 7 | 189 | 0.0046 | 0 | False |
| 8 | 96 | 0.0046 | 0 | False |

Removal refits preprocessing; these are pipeline sensitivities, not causal responsibility.

## Limits
- A single finite search is not a robustness certificate or a minimum-budget estimate.
- Surrogate (1 estimator, identity preprocessing) differs from the deployed estimator; only the Prove section describes deployed behaviour.
- Row influence is pipeline sensitivity, not causal responsibility; no fairness, medical or credit conclusions.