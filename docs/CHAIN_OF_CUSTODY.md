# Chain of Custody — decision stress testing (research module)

**Forensics for tabular AI, built on TabPFN-3.5.**
*Three rows flipped the prediction. Here is how we found them, proved it, and caught them.*

TabPFN does not train weights on your data; it *reads* the training rows in one forward pass, so
the training set is a differentiable input. Chain of Custody uses that to investigate one prediction:

| Step | Question | How |
|---|---|---|
| **1. Suspect** | Which few schema-valid rows, with adversarial labels, would flip this decision? | Gradient search through TabPFN's in-context learning |
| **2. Prove** | Does that exact row set flip the *deployed* model? | Refit standard TabPFN-3.5 and baselines; benign-label control |
| **3. Catch** | Can an auditor find the injected rows? | Exact leave-one-out refits (cheap: fitting is a forward pass) |

```python
from chainofcustody import Case

case = Case(X, y, feature_names=names, device="cuda:0")
finding  = case.suspect(x_query, k=3)   # gradient search on the differentiable surrogate
proof    = case.prove(finding)          # verified on standard TabPFN-3.5 + controls
suspects = case.catch(finding)          # exact leave-one-out ranking of every row
print(case.report(finding, proof, suspects))
```

```bash
custody investigate --dataset breast-cancer --seed 101 --target 0 --device cuda:0   # one command, writes .json + .md
custody app                                                                        # replay app (no GPU needed)
custody mcp                                                                        # evidence tools for an LLM agent
```

Example output (`results/custody/breast_101_0.md`): surrogate P 1.00 → 0.055; standard TabPFN-3.5
0.9998 → 0.0994 (flipped, benign-label control clean); the 3 appended rows are the top-3 of 203 by
leave-one-out influence. The HGB and logistic baselines did *not* flip on this case.

> Research prototype. Not a robustness certificate, minimum-budget estimate, fairness, causal,
> medical or credit assessment. Synthetic rows are schema-valid bounded perturbations, not
> realistic records. See [Limitations](#limitations).

## Key results (all generated from stored artifacts — see `docs/RESULTS.md`)

**1. Model internals: stable gradients through TabPFN-3.5 preprocessing.**
In tabpfn 9.1.0, `torch_nanstd` squares NaN-masked values before masking and differentiates
`sqrt` at zero variance: predictions are finite but backward passes through outlier clipping /
constant columns yield NaN. `auditkit/stability.py` changes only those backward conventions,
scoped to a context manager. On fresh context seeds: finite gradients **27/27 → 27/27**
(breast cancer), **9/27 → 27/27** (German credit), **0/27 → 27/27** (Taiwan), with **zero change
in any forward prediction** and gradient cosine ≈ 1 wherever the stock gradient was already finite.

**2. Pre-registered confirmation benchmark** (`docs/CONFIRMATION_PREREG.md`, written before the
run): 90 audits = 3 public datasets × 6 fresh context seeds × 5 targets, k = 3 appended rows.
Primary endpoint: verified flip of **standard TabPFN-3.5** with the exact rows and labels, over all
targets; comparator: the better of two query-matched black-box searches; Bonferroni 98.33% CIs.

| Dataset | Gradient search | Best matched black-box search | Paired difference [98.33% CI] | H1 |
|---|---:|---:|---|---|
| Breast cancer (30 numeric features) | **60%** | 13% | **+0.47 [+0.23, +0.73]** | **supported** |
| German credit (mixed / categorical) | 23% | 23% | +0.00 [−0.17, +0.17] | not supported |
| Taiwan default (mixed / categorical) | 33% | 30% | +0.03 [+0.00, +0.17] | not supported |

Gradient search through TabPFN finds substantially more verified counterexamples than
black-box search **on continuous numeric data**; on mixed categorical credit data, simple search
does about as well.

*Control:* for all 35 verified gradient flips of standard TabPFN (18 / 7 / 10), refitting the
same rows with **benign** labels flipped **none** — the effect comes from the adversarial labels.

*Second pre-registered test* (`docs/CONFIRMATION2_PREREG.md`): an exploratory development run
suggested 40 optimisation steps might help on mixed data. On fresh seeds 401–406 it did not
replicate: German credit 13% vs 10%, +0.03 [+0.00, +0.17]; Taiwan 27% vs 20%, +0.07
[+0.00, +0.23] (97.5% CIs) — H2 not supported on either dataset.

**3. Detection** (secondary): ranking all rows of the poisoned context by exact leave-one-out
influence on the target decision gives AUROC **0.994** (breast cancer), **0.896** (German credit),
**0.935** (Taiwan) for identifying the injected rows, ahead of out-of-fold label-suspicion
(0.988 / 0.722 / 0.871). Nearest-neighbour outlier scores fail (AUROC 0.01–0.15): the injected
rows look *more typical* than real rows — partly by construction, since search stays near real rows.

**4. Comparisons with other models** (descriptive; `docs/COMPARISONS_PREREG.md`, `docs/RESULTS.md` §4).
- *Catching:* exact leave-one-out with **TabPFN** finds the injected rows far better than leave-one-out with
  **trees** (AUROC 0.99 / 0.90 / 0.94 vs HGB 0.84 / 0.57 / 0.67 and random forest 0.63 / 0.56 / 0.52;
  breast cancer / German credit / Taiwan). A **logistic-regression** detector matches TabPFN's AUROC
  (0.99 / 0.88 / 0.94) at about 100× lower cost, so the case for TabPFN here is "as good as the best
  simple detector and much better than trees", not "uniquely best". Caveat: the rows were optimised against TabPFN.
- *Fragility:* with a 24-query black-box search aimed directly at each model, standard TabPFN flips less often
  than HGB on breast cancer (13% vs 53%) and is comparable elsewhere. TabPFN-crafted gradient rows flip
  standard TabPFN at 60% on breast cancer: the gradient is what finds those rows at that budget.
  Not a worst-case robustness comparison.

**5. Where TabPFN clearly wins — and where that does and doesn't help the audit.**
A pre-registered screen (`docs/REGIME_SCREEN_PREREG.md`; all 21 candidates listed in `docs/RESULTS.md` §5) found
tasks where TabPFN-3.5 beats logistic regression by a wide margin at context 200: eeg-eye-state
**0.846 vs 0.602** accuracy (random forest 0.720), MagicTelescope 0.845 vs 0.778, phoneme 0.813 vs 0.743,
mozilla4 0.925 vs 0.823. On seven of 21 candidates the gap was ≥ 5 points; on the rest there was none.
A second pre-registered test in those real regimes (120 audits, seeds 501–506):
- **Gradient search beat matched black-box search** (verified standard-TabPFN flips): eeg-eye-state
  **93% vs 43%** (+0.50 [+0.23, +0.77]) and MagicTelescope 67% vs 27% (+0.40 [+0.10, +0.70]) — supported.
  Phoneme (57% vs 37%, lower bound exactly 0.00) and mozilla4 (3% vs 7%) were **not** supported.
- **TabPFN leave-one-out did not beat logistic-regression leave-one-out** as a detector in any regime
  (paired AUROC difference +0.00, −0.02, −0.11, +0.03). It does beat tree-based leave-one-out
  (e.g. eeg-eye-state 0.93 vs HGB 0.65 / RF 0.70).
Net: where the task needs a nonlinear model, the accurate model is TabPFN — and it is also the one you can
differentiate through — but a cheap linear detector is as good at *catching* label-contradicting rows.

## What is built

| Component | Path |
|---|---|
| Audit engine: train-context scaling; hard range / one-hot / integer / immutable checks; Adam in standardized units; step-0 incumbent; frozen TabPFN weights; aborts (never masks) non-finite gradients | `auditkit/engine.py` |
| Forward-preserving backward stabilization (opt-in, version-checked) | `auditkit/stability.py` |
| Query-matched random and coordinate search baselines | `auditkit/search.py` |
| Receiver verification: standard TabPFN (fingerprint on/off), HGB, RF (default + regularized), logistic regression; refit/permutation/benign-label controls | `experiments/verify.py` |
| Exact leave-one-out sensitivity | `auditkit/influence.py` |
| CLI producing self-contained, hash-stamped JSON audit reports | `auditkit/cli.py` |
| Replay app (counterexample rows, constraints, traces, verification, detection) + benchmark page | `demo/app.py`, `demo/pages/1_Benchmark.py` |
| MCP server for an auditing agent | `auditkit/mcp_server.py` |
| Benchmarks, analyses, diagnostics | `experiments/` |
| Protocol, pre-registration, results, numerical notes | `docs/` |

## Quickstart

```bash
uv sync --extra demo --extra agent     # installs the `custody` command
uv run pytest -q tests
```

**Replay the app (no GPU, no model weights):**

```bash
uv run --extra demo streamlit run demo/app.py
```

Pick "Pre-registered confirmation (all 90, uncurated)" in the sidebar to browse every audit,
failures included, or open the **Benchmark** page.

**Generate a new audit (GPU; select the device explicitly):**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m auditkit.cli \
  --dataset breast-cancer --seed 101 --context 200 --target 0 --k 3 --steps 12 \
  --stable-backward --compare-search --device cuda:0 --out results/reports/my_audit.json
```

Model downloads require accepting the TabPFN-3.5 license (weights are not redistributed here).

**Agent / MCP** (replay tools need no model inference):

```bash
AUDIT_REPORT_DIR="$PWD/results/confirmation" AUDIT_BENCHMARK_DIR="$PWD/results/confirmation" \
  uv run --extra agent python -m auditkit.mcp_server
```

Tools: `list_audits`, `inspect_audit`, `inspect_counterexample`, `render_audit_report`,
`benchmark_summary`, `detection_findings`, `generate_demo_audit` (live generation only with
`AUDIT_ALLOW_INFERENCE=1`; public demo datasets only; serialized).

## Reproducing the paper-style results

```bash
# Numerical stabilization check (fresh seeds 201-203)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python experiments/check_stability.py
# Confirmation benchmark (one sequential process per listed GPU)
PYTHONPATH=. uv run python experiments/run_development.py --gpus 0,1,2,3 --targets 5 \
  --seeds 301,302,303,304,305,306 --stable-backward --outdir confirmation
uv run python experiments/summarize_paired.py results/confirmation --level 98.33
# Detection (shard across GPUs)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python experiments/detect_poison.py results/confirmation
uv run python experiments/summarize_detection.py results/confirmation/detection
uv run python experiments/make_results.py   # regenerates docs/RESULTS.md
```

Every report stores context row IDs and SHA256, target hash, the exact appended rows and labels,
configuration, dependency versions, per-method query counts, timings and failure flags.
`results/confirmation/protocol.json` records the source-file hashes of the frozen implementation.

## Limitations

- The gradient-search advantage was supported in 3 of 7 pre-registered dataset tests (breast cancer,
  eeg-eye-state, MagicTelescope), not on German credit or Taiwan (12 and 40 steps), phoneme or mozilla4.
- Dataset regimes for the second test were selected for a predictive TabPFN–logistic gap (disclosed).
- A logistic-regression leave-one-out detector matches TabPFN's at ~100× lower cost (H5 not supported).
- Bounds, one-hot validity and nearest-neighbour distance are geometric checks, not domain realism.
- Appended labels are adversarially corrupted (label-flip threat model).
- A failed search is not a robustness certificate; k = 3 is a fixed budget, not a minimum.
- The differentiable surrogate uses identity preprocessing and 1 estimator; it differs from the
  deployed estimator, which is why every counterexample is re-verified on standard TabPFN.
- Local gradients ignore ECDF rank discontinuities; they are search directions, not influence functions.
- Transfer results use rows optimized for TabPFN, not attacks optimized against other models.
- Public research datasets only; no clinical or lending conclusions.
- Early pilot scripts (`a_*`, `c1_*`–`c4_*`) are preserved but exploratory; a review found invalid
  constraints and unmatched baselines in them. Do not cite their numbers.

## Data and licensing

Original code: Apache-2.0 (`LICENSE`, `NOTICE`). This does not relicense data, third-party
packages or TabPFN weights. Datasets: Breast Cancer Wisconsin (Diagnostic), Statlog German
Credit, and Default of Credit Card Clients (Taiwan) — all UCI, CC BY 4.0.
