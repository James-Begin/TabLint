"""C1. Attribution: gradient-based / attention-based influence vs exact leave-one-out.

Usage: python c1_attribution.py <dataset> <n_ctx> <n_targets> <seed>
"""
import json
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from fz import core, data

name = sys.argv[1] if len(sys.argv) > 1 else "credit-g"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
T = int(sys.argv[3]) if len(sys.argv) > 3 else 50
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
dev = "cuda"
rng = np.random.default_rng(seed)

X, y = data.load(name, n_max=3000, seed=seed)
Xtr, Xte, ytr, yte = data.split(X, y, seed=seed)
idx = rng.choice(len(ytr), N, replace=False)
Xc, yc = Xtr[idx], ytr[idx]
Xt = Xte[rng.choice(len(yte), T, replace=False)]
clf = core.make_diff_clf(n_estimators=1, seed=seed)

Xc_t = torch.tensor(Xc, device=dev)
yc_t = torch.tensor(yc, device=dev, dtype=torch.float32)
Xt_t = torch.tensor(Xt, device=dev)


def P(Xa, ya, Xq=Xt_t):
    with torch.no_grad():
        return core.proba(clf, Xa, ya, Xq)[:, 1].detach().double().cpu().numpy()


with core.soft_labels(True):
    p0 = P(Xc_t, yc_t)
    p0b = P(Xc_t, yc_t)
    print("determinism max diff", float(np.abs(p0 - p0b).max()))

    # ---- exact LOO and exact label flip (shared over all targets) ----
    torch.cuda.synchronize(); t0 = time.time()
    loo = np.zeros((N, T))
    keep = torch.ones(N, dtype=torch.bool, device=dev)
    for i in range(N):
        keep[i] = False
        loo[i] = P(Xc_t[keep], yc_t[keep]) - p0
        keep[i] = True
    torch.cuda.synchronize(); t_loo = time.time() - t0
    t0 = time.time()
    flip = np.zeros((N, T))
    for i in range(N):
        yy = yc_t.clone(); yy[i] = 1 - yy[i]
        flip[i] = P(Xc_t, yy) - p0
    torch.cuda.synchronize(); t_flip = time.time() - t0

    # ---- gradients, one fwd+bwd per target ----
    gy = np.zeros((N, T)); gx = np.zeros((N, T)); gxd = np.zeros((N, T)); att = np.zeros((N, T))
    torch.cuda.synchronize(); t0 = time.time()
    for t in range(T):
        Xa = Xc_t.clone().requires_grad_(True)
        ya = yc_t.clone().requires_grad_(True)
        p = core.proba(clf, Xa, ya, Xt_t[t:t + 1])[0, 1]
        p.backward()
        gy[:, t] = ya.grad.double().cpu().numpy()
        g = Xa.grad.double().cpu().numpy()
        gx[:, t] = np.linalg.norm(g, axis=1)
        # first-order effect of moving row i onto the target point
        gxd[:, t] = (g * (Xt[t] - Xc)).sum(1)
    torch.cuda.synchronize(); t_grad = time.time() - t0
    # attention vote mass of the final decoder (one forward, all targets)
    core.CAPTURE["on"] = True
    with torch.no_grad():
        core.proba(clf, Xc_t, yc_t, Xt_t)
    A = core.CAPTURE["attn"][0].detach().double().cpu().numpy().T  # (N, T)
    core.CAPTURE["on"] = False
    print("attention shape", A.shape, "row-sum", float(A.sum(0).mean()))
    # sanity: attention-weighted vote vs p
    vote = (A * yc[:, None]).sum(0)
    print("corr(vote, p0)", float(np.corrcoef(vote, p0)[0, 1]))

# ---- proxies ----
sgn = (1 - 2 * yc)[:, None]              # flipping 0->1 is +, 1->0 is -
flip_pred = gy * sgn                     # linearised label-flip effect
resid = yc[:, None] - p0[None, :]
loo_att = -A * resid                     # removing a vote of mass a
mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
Z, Zt = (Xc - mu) / sd, (Xt - mu) / sd
d = np.sqrt(((Z[:, None, :] - Zt[None, :, :]) ** 2).sum(-1))  # (N,T)
sim = np.exp(-d / np.median(d))
loo_knn = -sim * resid

proxies = {
    "soft_label_grad(flip-linearised)": flip_pred,
    "decoder_attention": loo_att,
    "x_grad_dot(move-to-target)": gxd,
    "knn_similarity": loo_knn,
}
# LOO proxy from label gradient: removing a row of label y is approximated by
# moving its label half-way to the opposite class (removal ~ "neutral" label).
proxies["soft_label_grad*0.5 as LOO"] = -0.5 * gy * (2 * yc[:, None] - 1)
# magnitude-only proxies
mag = {"x_grad_norm": gx, "decoder_attention": A, "knn_similarity": sim,
       "abs_soft_label_grad": np.abs(gy)}


def per_target(score, truth, signed=True, k=10):
    rs, ov = [], []
    for t in range(T):
        s, tr = score[:, t], truth[:, t]
        if not signed:
            tr = np.abs(tr)
        rs.append(spearmanr(s, tr).correlation)
        top_true = set(np.argsort(-np.abs(truth[:, t]))[:k])
        top_s = set(np.argsort(-np.abs(s))[:k]) if signed else set(np.argsort(-s)[:k])
        ov.append(len(top_true & top_s) / k)
    return float(np.nanmean(rs)), float(np.mean(ov))


res = {"dataset": name, "N": N, "T": T, "seed": seed,
       "t_loo_s": t_loo, "t_flip_s": t_flip, "t_grad_s": t_grad,
       "loo_abs_mean": float(np.abs(loo).mean()), "loo_abs_max": float(np.abs(loo).max()),
       "flip_abs_mean": float(np.abs(flip).mean()),
       "spearman(flip,loo)": per_target(flip, loo)}
for nm, s in proxies.items():
    res[f"LOO signed | {nm}"] = per_target(s, loo)
res["FLIP signed | soft_label_grad"] = per_target(flip_pred, flip)
res["FLIP signed | decoder_attention"] = per_target(A * sgn, flip)
res["FLIP signed | knn"] = per_target(sim * sgn, flip)
for nm, s in mag.items():
    res[f"LOO |mag| | {nm}"] = per_target(s, loo, signed=False)
for k, v in res.items():
    print(k, v)
json.dump(res, open(f"results/c1_{name}_N{N}_T{T}_s{seed}.json", "w"), indent=1)
np.savez(f"results/c1_{name}_N{N}_T{T}_s{seed}.npz", loo=loo, flip=flip, gy=gy, gx=gx, A=A, sim=sim, p0=p0)
