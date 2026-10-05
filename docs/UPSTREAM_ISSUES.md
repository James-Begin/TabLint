# Upstream bug reports (drafts, not yet filed)

Found while building Proofread. Reproduced on 2026-10-03 with the latest PyPI releases: `tabpfn` 9.1.0 and
`tabpfn-extensions` 0.6.3. Machine: NVIDIA A10G, driver 580.178.04. Background, the full survey table and our
workaround are in `docs/KNOWN_ISSUES.md`. Reproduction scripts:
- `scripts/repro_tabpfn_extensions_cuda_dtype.py`: issue 1, minimal.
- `scripts/check_tabpfn_extensions_gpu.py`: all features, CPU vs CUDA.

GitHub searches of `PriorLabs/tabpfn-extensions` issues for "Index put" and "same device" returned no results on
2026-10-03.

---

## Issue 1 — `TabPFNUnsupervisedModel.outliers` and `tabpfn_crt` crash on CUDA (float32 targets vs float64 bar-distribution borders)

**Repository:** PriorLabs/tabpfn-extensions. The fix could also land in PriorLabs/TabPFN; see the end of this issue.

**Environment:** tabpfn 9.1.0, tabpfn-extensions 0.6.3, torch 2.14.1+cu130 (also reproduced the core PyTorch
behaviour with torch 2.4.1+cu121), Python 3.12, NVIDIA A10G.

**Reproduction** (the documented outlier example):
```python
from sklearn.datasets import load_breast_cancer
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

X = load_breast_cancer().data[:150, :4]
m = TabPFNUnsupervisedModel(tabpfn_clf=TabPFNClassifier(n_estimators=2), tabpfn_reg=TabPFNRegressor(n_estimators=2))
m.fit(X)
m.outliers(X, n_permutations=2)          # CUDA: RuntimeError; CPU: works
```
`tabpfn_extensions.pval_crt.tabpfn_crt(X, y, 0, B=5, K=5, device="cuda", model_version=ModelVersion.V3_5)` with a
continuous `y` fails the same way.

**Expected:** outlier scores / CRT result, as on CPU.

**Actual:**
```
RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Double for the source.
  tabpfn_extensions/unsupervised/unsupervised.py:783   -pred["criterion"].forward(logits_tensor, y_tensor)
  tabpfn/architectures/shared/bar_distribution.py:643  ignore_loss_mask = self.ignore_init(y)
  tabpfn/architectures/shared/bar_distribution.py:202  y[ignore_loss_mask] = self.borders[0]
```

**Cause:**
- `tabpfn` 9.x keeps bar-distribution `borders` in float64.
- The extension casts targets to `logits.dtype` (float32) before calling `criterion.forward`: in
  `unsupervised.py:775–779` (the comment says this is because MPS rejects float64) and in `pval_crt/utils.py:100–106`.
- `BarDistribution.ignore_init` then assigns the 0-dim float64 tensor `self.borders[0]` into the float32 `y` under a
  boolean mask.
- PyTorch silently casts this on CPU but raises on CUDA, **even when the mask is all False**. The same two-line
  behaviour occurs with torch 2.14.1/CUDA 13.0 and torch 2.4.1/CUDA 12.1:
  ```python
  y = torch.zeros(3, dtype=torch.float32, device="cuda"); b = torch.zeros(2, dtype=torch.float64, device="cuda")
  y[torch.isnan(y)] = b[0]     # RuntimeError on CUDA; fine on CPU
  ```

**Suggested fix** (either one):
- In TabPFN, `bar_distribution.py:202`: `y[ignore_loss_mask] = self.borders[0].to(y.dtype)`. This fixes every caller.
- In the extension: cast `y` to `criterion.borders.dtype` (and the logits too, if needed) except on MPS.

**Workaround:** wrap the regressor so that `predict(..., output_type="full")` returns `logits.double()`. The
extension then casts targets to float64 and the assignment succeeds.

---

## Issue 2 — `TabPFNUnsupervisedModel.generate_synthetic_data` crashes on CUDA (device mismatch)

**Repository:** PriorLabs/tabpfn-extensions

**Reproduction:**
```python
import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

X = np.random.default_rng(0).normal(size=(60, 4))
reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device="cuda", n_estimators=2)
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device="cuda", n_estimators=2)
m = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg); m.fit(X)
m.generate_synthetic_data(n_samples=5, n_permutations=1)   # CUDA: RuntimeError; CPU: works
```

**Actual:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
  tabpfn_extensions/unsupervised/unsupervised.py:490
      impute_X[torch.isnan(y_predict), column_idx] = pred_sampled.to(y_predict.dtype)
```

**Likely cause** (inferred from the code, not stepped through): `pred_sampled` comes from the model on CUDA, while
`impute_X` is a CPU copy of the input. `.to(y_predict.dtype)` changes the dtype but not the device.

**Suggested fix:** `pred_sampled.to(device=impute_X.device, dtype=y_predict.dtype)`.

**Note:** `impute(...)` worked on CUDA in our test, and so did survival, hurdle, `cp_missing_data` and Bayesian
optimisation (with inputs on the model's device).

---

## Not bugs (checked)

- `bayesian_optimization.propose_next_point` raised `indices should be either on cpu or on the same device` only
  because we passed CPU tensors to a CUDA model. With inputs on the model's device it works. A clearer error message
  or a docstring note would help, but this is not a defect.
- `survival.SurvivalTabPFN` needs the optional `scikit-survival` package. With it installed, it works on CPU and CUDA.
