"""Diagnostics: row-permutation noise floor; soft-label gradient vs finite differences."""
import numpy as np
import torch
from scipy.stats import spearmanr

from fz import core, data

dev = "cuda"
rng = np.random.default_rng(0)
X, y = data.load("credit-g")
Xtr, Xte, ytr, yte = data.split(X, y, seed=0)
idx = rng.choice(len(ytr), 300, replace=False)
Xc = torch.tensor(Xtr[idx], device=dev); yc = torch.tensor(ytr[idx], device=dev, dtype=torch.float32)
Xt = torch.tensor(Xte[:30], device=dev)
clf = core.make_diff_clf(n_estimators=1, seed=0)


def P(Xa, ya):
    with torch.no_grad():
        return core.proba(clf, Xa, ya, Xt)[:, 1].detach().double().cpu().numpy()


with core.soft_labels(True):
    p0 = P(Xc, yc)
    for r in range(3):
        perm = torch.tensor(rng.permutation(300), device=dev)
        print("perm noise |dp| mean/max", np.abs(P(Xc[perm], yc[perm]) - p0).mean(), np.abs(P(Xc[perm], yc[perm]) - p0).max())
    # duplicate-row (append exact copy of row 0) effect
    # gradient vs FD for target 0
    ya = yc.clone().requires_grad_(True)
    p = core.proba(clf, Xc, ya, Xt[:1])[0, 1]; p.backward()
    g = ya.grad.double().cpu().numpy()
    sg = 1 - 2 * ytr[idx]
    for eps in [1e-3, 0.05, 0.25, 0.5, 1.0]:
        fd = []
        for i in range(60):
            yy = yc.clone(); yy[i] = yy[i] + eps * sg[i]
            fd.append((P(Xc, yy)[0] - p0[0]) / eps)
        fd = np.array(fd)
        print(f"eps={eps}: spearman(grad*sgn, FD)={spearmanr(g[:60]*sg[:60], fd).correlation:.3f}  "
              f"mean|FD|={np.abs(fd).mean():.4f} mean|grad|={np.abs(g[:60]).mean():.4f}")
