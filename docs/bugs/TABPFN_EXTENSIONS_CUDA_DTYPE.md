# Bug report: CUDA dtype mismatch in TabPFN extension outlier scoring

**Suggested issue title:** `outliers() crashes on CUDA: float32 targets vs float64 BarDistribution borders`

**Status:** Written up in this repository; not submitted to the upstream issue tracker. The affected versions and GPU observations below were recorded on 2026-10-03. This report was prepared on 2026-10-05; CUDA was not available on the documentation machine, so the GPU reproduction was not rerun today. No claim is made about later package versions.

## Summary and impact

With `tabpfn==9.1.0` and `tabpfn-extensions==0.6.3`, calling `TabPFNUnsupervisedModel.outliers()` with a CUDA regressor raises a dtype mismatch inside `BarDistribution.ignore_init`. The recorded CPU control completes successfully. The failure can occur with entirely finite input data.

This blocked the optional TabPFN extension baseline during TabLint's benchmark work. TabLint's normal numerical checks use TabPFN directly and call the predictive distribution's `cdf` method; they do not call the extension's failing `outliers()` path. The extension is an optional benchmark dependency, not a requirement for normal TabLint setup.

The same dtype failure was recorded in the extension's conditional randomization test. A separate synthetic-data device mismatch is outside this report; see the [feature survey](../KNOWN_ISSUES.md#survey-other-tabpfn-extensions-features-on-gpu).

## Recorded environment

| Component | Version or observation |
| --- | --- |
| TabPFN | 9.1.0 |
| tabpfn-extensions | 0.6.3 |
| Model used in the feature survey | TabPFN-3.5 |
| PyTorch / CUDA, original tensor reproduction | 2.14.1 / 13.0 |
| PyTorch / CUDA, second tensor reproduction | 2.4.1 / 12.1 |
| NVIDIA driver | 580.178.04, same GPU for both tensor checks |
| CPU control | Masked assignment and extension outlier example succeed |
| CUDA result | Both tested PyTorch environments reject the masked assignment |

The second environment checks the underlying tensor operation, not a second complete extension installation. Full hardware and OS metadata were not preserved with the original notes. [Original observations](../KNOWN_ISSUES.md).

## Reproduction

From a TabLint checkout, on a machine with a CUDA-capable PyTorch installation:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --extra baselines python scripts/repro_tabpfn_extensions_cuda_dtype.py
```

The [reproduction script](../../scripts/repro_tabpfn_extensions_cuda_dtype.py) prints PyTorch/CUDA availability, checks the masked assignment on CPU and CUDA, then runs an extension example on 150 rows and four features from scikit-learn's breast-cancer dataset. It contains no TabLint inference code. Verify that its output says `cuda available: True`; otherwise it is only a CPU control. Model weights and first-use model access are needed for the extension example.

Run the CPU control separately:

```sh
CUDA_VISIBLE_DEVICES="" uv run --extra baselines python scripts/repro_tabpfn_extensions_cuda_dtype.py
```

The underlying operation can also be reproduced without model weights or `tabpfn-extensions`:

```python
import torch

for device in ["cpu", "cuda"]:
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda: skipped (unavailable)")
        continue
    y = torch.zeros(3, dtype=torch.float32, device=device)
    borders = torch.zeros(2, dtype=torch.float64, device=device)
    try:
        y[torch.isnan(y)] = borders[0]
        print(f"{device}: success")
    except RuntimeError as error:
        print(f"{device}: {error}")
```

The mask is empty in this example, demonstrating that missing values are not required to trigger the dtype check.

## Expected and actual behavior

**Expected:** Outlier scoring completes on CUDA and returns density-based scores, as it does in the CPU control. Replacing ignored targets with a dummy boundary should respect the target tensor's dtype and device.

**Actual:** The recorded CUDA run raises:

```text
RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Double for the source
```

The failure point is `tabpfn/architectures/shared/bar_distribution.py:202`, in `BarDistribution.ignore_init`. This is the recorded exception and location; a complete original traceback was not retained.

## Root cause

The extension converts regression targets to the logits' dtype and device before invoking the distribution as a loss. In the failing run, those targets are float32 and the distribution borders are float64. [Extension source, v0.6.3](https://github.com/PriorLabs/tabpfn-extensions/blob/v0.6.3/src/tabpfn_extensions/unsupervised/unsupervised.py).

`ignore_init` performs a masked assignment from a boundary tensor before the later bucket-mapping code converts targets to the borders' dtype. CUDA rejects this mixed-dtype assignment; the recorded CPU control casts the scalar. The failure was reproduced under two PyTorch/CUDA combinations, so changing CUDA versions alone did not resolve the tensor-level reproduction. [Distribution source, v9.1.0](https://github.com/PriorLabs/TabPFN/blob/v9.1.0/src/tabpfn/architectures/shared/bar_distribution.py#L195-L204).

## Proposed upstream fix

Convert the dummy value to the destination tensor's dtype and device at the assignment:

```diff
- y[ignore_loss_mask] = self.borders[0]
+ y[ignore_loss_mask] = self.borders[0].to(device=y.device, dtype=y.dtype)
```

This is a proposed patch, not an upstream merged fix. It preserves the target dtype and avoids requiring float64 support on Apple MPS. It does not change the stored distribution boundaries. GPU and MPS validation remains necessary before merging upstream.

Suggested regression checks:

- CPU and CUDA with float32 targets and float64 borders, using both an empty mask and actual NaN targets.
- Same-dtype float32 and float64 controls; finite targets remain unchanged, and ignored-target masks remain correct.
- An end-to-end CUDA `outliers()` call and the conditional randomization test.
- An MPS float32 control where available, to preserve the extension's reason for casting targets.

## Workaround used in TabLint's benchmark

The optional baseline wrapper converts the returned logits to float64, causing the extension to convert its targets to float64 too. This makes their dtype match the borders. See [`ext_regressor`](../../benchmarks/proofread_addendum.py#L21-L36).

The upcast preserves the represented logits but can change subsequent density calculations through increased arithmetic precision. It is not the same operation as the CPU scalar cast, and no universal bound on score differences is asserted. The benchmark method and settings were retained; the compatibility adjustment is disclosed in the [pre-registration amendment](../PROOFREAD_ADDENDUM_PREREG.md#amendment-before-any-result-compatibility-shim) and [known-issue corrections](../KNOWN_ISSUES.md).

The pinned source and tensor operation were checked while preparing this report. The CPU control succeeds locally; CUDA is unavailable here. The proposed patch has not been validated on CUDA or MPS during this documentation update.
