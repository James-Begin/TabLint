"""Minimal reproduction: tabpfn-extensions outlier detection crashes on CUDA with tabpfn 9.x.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --extra baselines python scripts/repro_tabpfn_extensions_cuda_dtype.py   # crashes
    CUDA_VISIBLE_DEVICES=""  uv run --extra baselines python scripts/repro_tabpfn_extensions_cuda_dtype.py  # works
See docs/bugs/TABPFN_EXTENSIONS_CUDA_DTYPE.md.
"""
import torch

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}); cuda available: {torch.cuda.is_available()}")

# 1. The underlying PyTorch behaviour: masked assignment of a 0-dim float64 tensor into a float32 tensor.
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    y = torch.zeros(3, dtype=torch.float32, device=dev)
    borders = torch.zeros(2, dtype=torch.float64, device=dev)
    try:
        y[torch.isnan(y)] = borders[0]          # what tabpfn's BarDistribution.ignore_init does
        print(f"[{dev}] float32[mask] = float64 0-dim tensor -> ok (silently cast)")
    except RuntimeError as e:
        print(f"[{dev}] float32[mask] = float64 0-dim tensor -> RuntimeError: {str(e)[:80]}")

# 2. The documented tabpfn-extensions usage, unmodified.
from sklearn.datasets import load_breast_cancer
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

X = load_breast_cancer().data[:150, :4]
model = TabPFNUnsupervisedModel(tabpfn_clf=TabPFNClassifier(n_estimators=2), tabpfn_reg=TabPFNRegressor(n_estimators=2))
model.fit(X)
try:
    scores = model.outliers(X, n_permutations=2)
    print("TabPFNUnsupervisedModel.outliers -> ok", scores[:3])
except RuntimeError as e:
    print("TabPFNUnsupervisedModel.outliers -> RuntimeError:", str(e)[:100])
