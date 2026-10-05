"""Opt-in backward stabilization for TabPFN 9.1.0 preprocessing (experimental).

Installed `tabpfn/preprocessing/torch/ops.py::torch_nanstd` squares NaN-masked
differences before masking, and takes sqrt(0) on zero-variance columns. Forward
values are finite but backward can produce NaN. This replacement keeps the forward
computation and changes only those backward conventions:
  * NaN-masked entries are zeroed BEFORE squaring (forward identical: masked anyway)
  * d sqrt(v)/dv is defined as 0 at v == 0 (forward identical)
The patch is scoped to a context manager, is not thread-safe across concurrent
model calls, and restores the original bindings on exit. It is version-checked.
"""
from __future__ import annotations

import contextlib
import threading

_LOCK = threading.RLock()
SUPPORTED_TABPFN = {"9.1.0"}


def _safe_functions():
    import torch

    class SafeSqrt(torch.autograd.Function):
        @staticmethod
        def forward(ctx, variance):
            root = torch.sqrt(variance)
            ctx.save_for_backward(root)
            return root

        @staticmethod
        def backward(ctx, incoming):
            root, = ctx.saved_tensors
            positive = root > 0
            divisor = torch.where(positive, 2 * root, torch.ones_like(root))
            return incoming * torch.where(positive, 1 / divisor, torch.zeros_like(root))

    def safe_nanstd(x, axis=0):
        nan_mask = torch.isnan(x)
        num_valid = torch.where(nan_mask, torch.zeros_like(x), torch.ones_like(x)).sum(dim=axis)
        value_sum = torch.where(nan_mask, torch.zeros_like(x), x).sum(dim=axis)
        mean = value_sum / num_valid.clamp(min=1.0)
        diff = torch.where(nan_mask, torch.zeros_like(x), x - mean.unsqueeze(axis).expand_as(x))
        variance = torch.square(diff).sum(dim=axis) / (num_valid - 1).clamp(min=1.0)
        return SafeSqrt.apply(variance)

    return safe_nanstd


@contextlib.contextmanager
def stable_backward(enabled: bool = True):
    """Patch TabPFN's nan-std backward inside this block only."""
    if not enabled:
        yield False
        return
    import importlib.metadata
    version = importlib.metadata.version("tabpfn")
    if version not in SUPPORTED_TABPFN:
        raise RuntimeError(f"stable_backward validated only for tabpfn {SUPPORTED_TABPFN}, found {version}")
    import tabpfn.preprocessing.torch.torch_soft_clip_outliers as outlier_module
    import tabpfn.preprocessing.torch.torch_standard_scaler as scaler_module
    safe = _safe_functions()
    with _LOCK:
        original_s, original_o = scaler_module.torch_nanstd, outlier_module.torch_nanstd
        scaler_module.torch_nanstd = safe
        outlier_module.torch_nanstd = safe
        try:
            yield True
        finally:
            scaler_module.torch_nanstd = original_s
            outlier_module.torch_nanstd = original_o
