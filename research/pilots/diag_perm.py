"""Diagnostics: sources of prediction noise under row permutation / row removal."""
import numpy as np
import torch

from fz import core, data

dev = "cuda"
rng = np.random.default_rng(0)
X, y = data.load("credit-g")
Xtr, Xte, ytr, yte = data.split(X, y, seed=0)
idx = rng.choice(len(ytr), 300, replace=False)
Xc_np, yc_np = Xtr[idx], ytr[idx]
Xc = torch.tensor(Xc_np, device=dev); yc = torch.tensor(yc_np, device=dev, dtype=torch.float32)
Xt = torch.tensor(Xte[:30], device=dev)
perms = [rng.permutation(300) for _ in range(3)]

for label, ic in [("default", None), ("no_fingerprint", {"FINGERPRINT_FEATURE": False}),
                  ("no_fp_no_shift", {"FINGERPRINT_FEATURE": False, "FEATURE_SHIFT_METHOD": None}),
                  ("no_fp_no_shift_no_outlier", {"FINGERPRINT_FEATURE": False, "FEATURE_SHIFT_METHOD": None,
                                                 "OUTLIER_REMOVAL_STD": None})]:
    try:
        clf = core.make_diff_clf(n_estimators=1, seed=0)
        if ic:
            clf.inference_config = ic
        def P(Xa, ya):
            with torch.no_grad():
                return core.proba(clf, Xa, ya, Xt)[:, 1].detach().double().cpu().numpy()
        p0 = P(Xc, yc)
        pn = [np.abs(P(Xc[torch.tensor(pm, device=dev)], yc[torch.tensor(pm, device=dev)]) - p0).mean() for pm in perms]
        # duplicate-free "removal" of a row with tiny effect: drop row then re-add identical copy at end
        loo = [np.abs(P(Xc[1:], yc[1:]) - p0).mean(), np.abs(P(Xc[:-1], yc[:-1]) - p0).mean()]
        print(f"[diff {label}] perm noise mean|dp| {np.round(pn,4)}  LOO(row0,row-1) mean|dp| {np.round(loo,4)}")
    except Exception as e:
        print(label, "ERROR", repr(e)[:300])

# standard sklearn-style model
for label, kw in [("std default", {}), ("std no_fp", {"inference_config": {"FINGERPRINT_FEATURE": False}})]:
    c = core.make_std_clf(n_estimators=1, **kw)
    c.fit(Xc_np, yc_np); p0 = c.predict_proba(Xte[:30])[:, 1]
    pn = []
    for pm in perms:
        c.fit(Xc_np[pm], yc_np[pm]); pn.append(np.abs(c.predict_proba(Xte[:30])[:, 1] - p0).mean())
    c.fit(Xc_np[1:], yc_np[1:]); l = np.abs(c.predict_proba(Xte[:30])[:, 1] - p0).mean()
    print(f"[{label}] perm noise {np.round(pn,4)} LOO row0 {l:.4f}")
