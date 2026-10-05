# Chain of Custody — investigation report

Threat model: k rows are appended to the training context with adversarially chosen labels; feature values are schema-valid and bounded near real rows (geometric checks, not domain realism).

## 1. Suspect (differentiable surrogate; not yet verified)
- Query decision on surrogate: P(class 1) = 0.9999 → class 1
- After appending 3 rows: P = 0.0554 → class 0; flip=True, constraints valid=True
- Appended labels: [0, 0, 0]

| Feature | Query | Row 1 | Row 2 | Row 3 |
|---|---:|---:|---:|---:|
| mean symmetry | 0.2152 | 0.1981 | 0.2041 | 0.1665 |
| area error | 134.8 | 113.6 | 210.4 | 167.1 |
| symmetry error | 0.02591 | 0.02232 | 0.01564 | 0.01624 |
| compactness error | 0.05839 | 0.03844 | 0.04752 | 0.0353 |
| mean compactness | 0.2146 | 0.1657 | 0.1522 | 0.1672 |
| mean fractal dimension | 0.06673 | 0.05831 | 0.069 | 0.06412 |
| radius error | 0.9806 | 0.6943 | 1.316 | 1.028 |
| smoothness error | 0.00794 | 0.006708 | 0.006859 | 0.004831 |

## 2. Prove (refit on the exact rows and labels)
| Receiver | P before | P after | Decision flipped | Benign-label control flipped |
|---|---:|---:|---|---|
| tabpfn_standard | 0.9998 | 0.0994 | True | False |
| hgb | 0.9967 | 0.7303 | False | False |
| logistic | 1.0000 | 0.7782 | False | False |

**Verified on deployed TabPFN-3.5 (flip and benign-label control clean): True**

## 3. Catch (exact leave-one-out, context + appended rows)
- Appended rows ranked in the top 3: **3/3**

| Rank | Row | |Δp| | Label | Appended |
|---:|---:|---:|---:|---|
| 1 | 200 | 0.6065 | 0 | True |
| 2 | 201 | 0.2101 | 0 | True |
| 3 | 202 | 0.1640 | 0 | True |
| 4 | 6 | 0.1144 | 1 | False |
| 5 | 178 | 0.0886 | 1 | False |
| 6 | 107 | 0.0591 | 1 | False |
| 7 | 0 | 0.0523 | 0 | False |
| 8 | 198 | 0.0476 | 1 | False |

Removal refits preprocessing; these are pipeline sensitivities, not causal responsibility.

## Limits
- A single finite search is not a robustness certificate or a minimum-budget estimate.
- Surrogate (1 estimator, identity preprocessing) differs from the deployed estimator; only the Prove section describes deployed behaviour.
- Row influence is pipeline sensitivity, not causal responsibility; no fairness, medical or credit conclusions.