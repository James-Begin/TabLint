# Known issues

## tabpfn-extensions outlier detection crashes on GPU with tabpfn 9.x

[Full bug report: reproduction, expected behavior, environment and proposed fix](bugs/TABPFN_EXTENSIONS_CUDA_DTYPE.md). Prepared for upstream reporting; not submitted upstream.

**Symptom.** `TabPFNUnsupervisedModel.outliers(...)` from `tabpfn-extensions` raises
`RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Double for the source`
whenever the TabPFN regressor runs on a CUDA GPU. The same code runs fine on CPU.

**Not caused by this project.** It reproduces with Prior Labs' documented example (breast-cancer data, default models)
in a clean script with none of our code: `scripts/repro_tabpfn_extensions_cuda_dtype.py`.

| Environment | Documented example on CPU | Documented example on CUDA |
|---|---|---|
| tabpfn 9.1.0 + tabpfn-extensions 0.6.3 (latest on PyPI, 2026-10-03) | works | **crashes** |

The extension declares `tabpfn>=8.4.0`, so this is a supported combination.

**Root cause.** Two reasonable choices collide:
1. `tabpfn` 9.x stores the regressor's bar-distribution borders as **float64**
   (`tabpfn/architectures/shared/bar_distribution.py`: "float64 borders keep float64 results").
2. `tabpfn-extensions` casts targets to the **logits' dtype (float32)** before calling `criterion.forward`
   (`tabpfn_extensions/unsupervised/unsupervised.py`; its comment says this is because Apple MPS rejects float64).

`BarDistribution.ignore_init` then runs `y[torch.isnan(y)] = self.borders[0]`. PyTorch silently casts the 0-dim float64
value on CPU but raises on CUDA, **even when no element is NaN**.

**Not a CUDA or driver version issue.** The two-line PyTorch check behaves identically under torch 2.14.1 / CUDA 13.0 and
torch 2.4.1 / CUDA 12.1, on the same GPU (driver 580.178.04): it works on CPU and raises on CUDA. This is long-standing
PyTorch behaviour, not something specific to our setup.

**Workaround used here** (`benchmarks/proofread_addendum.py`, `ext_regressor`): the baseline regressor's `predict(...,
output_type="full")` returns logits upcast to float64. The extension then casts targets to float64 too, and the
assignment succeeds. This only changes the floating-point precision of the baseline's density computation (float64
instead of float32); the method and its settings are unchanged. This differs from the CPU control, which casts the
boundary scalar to the destination dtype during assignment.

**Possible upstream fix** (either package): cast `self.borders[0]` to `y.dtype` in `ignore_init`, e.g.
`y[mask] = self.borders[0].to(y.dtype)`, or have the extension cast logits and targets to the borders' dtype except on MPS.

**Corrections to earlier wording.**
- The amendment in `docs/PROOFREAD_ADDENDUM_PREREG.md` says the failure happens "with tabpfn 9.1.0". It happens only on
  CUDA.
- It also says "no values are changed". More precisely, the baseline is computed at float64 rather than float32
  precision. The represented logits are preserved, but downstream density arithmetic can change; no universal bound
  on score differences has been established.
- The original pre-run version is preserved in Git history; its hash and the later dataset-scope correction are recorded in `docs/BENCHMARK_AMENDMENT.md`. The separate dtype-wording corrections remain recorded here.


## Survey: other tabpfn-extensions features on GPU

We ran every tabpfn-extensions 0.6.3 feature that uses TabPFN's regression distribution, on CUDA and on CPU, with
tabpfn 9.1.0 and TabPFN-3.5 models, using small synthetic inputs placed on the model's device:
`scripts/check_tabpfn_extensions_gpu.py`. All features work on CPU.

| Feature | CUDA | Failure point | Cause |
|---|---|---|---|
| `unsupervised.TabPFNUnsupervisedModel.outliers` | **crashes** | `tabpfn/.../bar_distribution.py:202` | float32 targets vs float64 borders (above) |
| `pval_crt.tabpfn_crt` (conditional randomization test) | **crashes** | `tabpfn/.../bar_distribution.py:202` | Same: `pval_crt/utils.py:100–106` casts `y` to `logits.dtype` and calls `criterion(logits, y)` |
| `unsupervised.TabPFNUnsupervisedModel.generate_synthetic_data` | **crashes** | `tabpfn_extensions/unsupervised/unsupervised.py:490` | Device mismatch. Sampled values (on CUDA) are written into the CPU imputation tensor, and `.to(y_predict.dtype)` changes the dtype but not the device. Inferred from the code line, not stepped through. |
| `unsupervised.TabPFNUnsupervisedModel.impute` | works | | |
| `survival.SurvivalTabPFN` (needs `scikit-survival`) | works | | Uses `criterion.cdf`, which promotes dtypes via `torch.where` |
| `bayesian_optimization.propose_next_point` | works | | Inputs must be on the model's device. CPU tensors with a CUDA model raise at `bo.py:105`; that's our misuse, not a bug. |
| `hurdle.AutoHurdleRegressor` | works | | |
| `cp_missing_data.CPMDATabPFNRegressor.fit` | works | | |

The two dtype crashes share one root cause and one fix (cast `self.borders[0]` to `y.dtype` in
`BarDistribution.ignore_init`). The synthetic-data crash is separate: it would need `.to(impute_X.device, y_predict.dtype)`
at `unsupervised.py:490`.

**Status.** As of 2026-10-03, GitHub searches of `PriorLabs/tabpfn-extensions` issues for "Index put" and "same device"
return no results, so these appear unreported. A related bug in the same density code (#307, the chain-rule term used
the wrong feature) was fixed in June 2026. We have not filed reports.
