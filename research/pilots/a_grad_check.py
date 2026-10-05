"""A. Minimal gradient feasibility check for TabPFN-3.5 (differentiable_input)."""
import time

import numpy as np
import torch

from fz import core, data

dev = "cuda"
X, y = data.load("credit-g")
Xtr, Xte, ytr, yte = data.split(X, y, seed=0)
Xtr, ytr = Xtr[:500], ytr[:500]
print("credit-g one-hot shape", X.shape, "train", Xtr.shape)

# 1) standard (non-differentiable) model for reference
std = core.make_std_clf(n_estimators=1)
std.fit(Xtr, ytr)
p_std = std.predict_proba(Xte[:20])[:, 1]

# 2) differentiable model
clf = core.make_diff_clf(n_estimators=1)
Xtr_t = torch.tensor(Xtr, device=dev, requires_grad=True)
ytr_t = torch.tensor(ytr, device=dev, dtype=torch.float32, requires_grad=True)
Xte_t = torch.tensor(Xte[:20], device=dev, requires_grad=True)
t = time.time()
p = core.proba(clf, Xtr_t, ytr_t, Xte_t)
print("diff forward ok, out shape", tuple(p.shape), "time", round(time.time() - t, 3))
p1 = p[:, 1]
print("std vs diff p1 max abs diff (different preprocessing!)",
      float(np.abs(p1.detach().cpu().numpy() - p_std).max()))
p1[0].backward()
print("grad X_test  :", None if Xte_t.grad is None else float(Xte_t.grad.abs().sum()))
print("grad X_train :", None if Xtr_t.grad is None else float(Xtr_t.grad.abs().sum()),
      "nonzero rows", None if Xtr_t.grad is None else int((Xtr_t.grad.abs().sum(1) > 0).sum()))
print("grad y_train (hard labels):", None if ytr_t.grad is None else float(ytr_t.grad.abs().sum()))

# 3) soft-label patch
with core.soft_labels(True):
    Xtr_t.grad = None
    ytr_t.grad = None
    p_soft = core.proba(clf, Xtr_t, ytr_t, Xte_t)
    print("soft vs hard at integer labels, max abs diff:",
          float((p_soft - p).abs().max()))
    p_soft[0, 1].backward()
    g = ytr_t.grad
    print("grad y_train (soft labels):", None if g is None else float(g.abs().sum()))
    # finite-difference check of label gradient on 5 rows: flip label fully
    gy = g.detach().cpu().numpy()
    base = float(p_soft[0, 1])
    for i in np.argsort(-np.abs(gy))[:5]:
        yy = ytr_t.detach().clone()
        yy[i] = 1 - yy[i]
        with torch.no_grad():
            pf = float(core.proba(clf, Xtr_t.detach(), yy, Xte_t.detach())[0, 1])
        print(f"  row {i}: lin-pred d={gy[i]*(1-2*ytr[i]):+.5f}  exact flip d={pf-base:+.5f}")

# 4) n_estimators>1 and timing
for ne in [1, 4, 8]:
    c = core.make_diff_clf(n_estimators=ne)
    Xa = torch.tensor(Xtr, device=dev, requires_grad=True)
    torch.cuda.synchronize(); t = time.time()
    pp = core.proba(c, Xa, torch.tensor(ytr, device=dev, dtype=torch.float32), Xte_t.detach())
    pp[:, 1].sum().backward(); torch.cuda.synchronize()
    print(f"n_est={ne}: fwd+bwd {time.time()-t:.3f}s grad-nonzero={bool(Xa.grad.abs().sum()>0)} "
          f"mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
