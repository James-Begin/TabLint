"""Smoke-test tabpfn-extensions features that use TabPFN's regression distribution, on the current device.

Usage: CUDA_VISIBLE_DEVICES=0 uv run --extra baselines --with scikit-survival python scripts/check_tabpfn_extensions_gpu.py
       CUDA_VISIBLE_DEVICES="" ... (CPU). See docs/KNOWN_ISSUES.md."""
import os, sys, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
X = rng.normal(size=(120, 4)); y = X[:, 0] * 2 + X[:, 1] ** 2 + rng.normal(0, .3, 120)
from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabpfn.constants import ModelVersion
R = lambda **k: TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device=dev, n_estimators=2, **k)
C = lambda: TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device=dev, n_estimators=2)

def t_outliers():
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
    m = TabPFNUnsupervisedModel(tabpfn_clf=C(), tabpfn_reg=R()); m.fit(X[:60]); m.outliers(X[:60], n_permutations=1)
def t_impute():
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
    Xm = X[:60].copy(); Xm[3, 1] = np.nan
    m = TabPFNUnsupervisedModel(tabpfn_clf=C(), tabpfn_reg=R()); m.fit(Xm); m.impute(torch.tensor(Xm, dtype=torch.float32), n_permutations=1)
def t_synthetic():
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
    m = TabPFNUnsupervisedModel(tabpfn_clf=C(), tabpfn_reg=R()); m.fit(X[:60]); m.generate_synthetic_data(n_samples=5, n_permutations=1)
def t_crt():
    from tabpfn_extensions.pval_crt import tabpfn_crt
    tabpfn_crt(X, y, 0, B=5, K=5, device=dev, model_version=ModelVersion.V3_5)
def t_survival():
    from tabpfn_extensions.survival.survival import SurvivalTabPFN
    ev = rng.random(120) < .7; tm = np.abs(rng.normal(5, 2, 120))
    yy = np.array(list(zip(ev, tm)), dtype=[("event", bool), ("time", float)])
    s = SurvivalTabPFN(cls_model=C(), reg_model=R()); s.fit(X, yy); s.predict_cif_at(X[:5], [3.0, 5.0])
def t_bo():
    from tabpfn_extensions.bayesian_optimization import propose_next_point
    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device=dev, n_estimators=1, differentiable_input=True)
    tx = torch.rand(20, 3, device=dev); ty = (tx ** 2).sum(1)   # inputs on the model device (correct usage)
    propose_next_point(reg, tx, ty, n_candidates=32, top_k=2, n_refine_steps=2)
def t_hurdle():
    from tabpfn_extensions.hurdle.hurdle import AutoHurdleRegressor
    yz = np.where(rng.random(120) < .4, 0, np.abs(y)); AutoHurdleRegressor(classifier=C(), regressor=R()).fit(X, yz).predict(X[:5])
def t_cpmda():
    from tabpfn_extensions.cp_missing_data.cp_missing_data import CPMDATabPFNRegressor
    Xm = X.copy(); Xm[::7, 2] = np.nan
    CPMDATabPFNRegressor(tabpfn_estimator=R(), seed=0).fit(Xm, y)

for name, fn in [("unsupervised.outliers", t_outliers), ("unsupervised.impute", t_impute), ("unsupervised.generate_synthetic_data", t_synthetic),
                 ("pval_crt.tabpfn_crt", t_crt), ("survival.SurvivalTabPFN", t_survival), ("bayesian_optimization.propose_next_point", t_bo),
                 ("hurdle.AutoHurdleRegressor", t_hurdle), ("cp_missing_data.CPMDATabPFNRegressor.fit", t_cpmda)]:
    try:
        fn(); print(f"[{dev}] {name}: ok", flush=True)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"[{dev}] {name}: {type(e).__name__}: {str(e)[:90]}  @ {tb.filename.split('site-packages/')[-1]}:{tb.lineno}", flush=True)
