"""C3. Fragility / poisoning audit: smallest number k of synthetic context rows that
flips a decision, found by gradient optimisation of the appended rows' features.

Usage: python c3_poison.py <dataset> <n_ctx> <n_targets> <shard> <n_shards> [steps]
Writes results/c3_<dataset>_shard<i>.jsonl (one line per target x k x method).
"""
import json
import sys
import time

import numpy as np
import torch

from fz import core, data

name = sys.argv[1]
N, T = int(sys.argv[2]), int(sys.argv[3])
shard, n_shards = int(sys.argv[4]), int(sys.argv[5])
STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 40
LR = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
KS = [1, 3, 5]
dev = "cuda"
seed = 0
rng = np.random.default_rng(seed)

X, y = data.load(name, n_max=3000, seed=seed)
Xtr, Xte, ytr, yte = data.split(X, y, seed=seed)
idx = rng.choice(len(ytr), N, replace=False)
Xc, yc = Xtr[idx], ytr[idx]
tsel = rng.choice(len(yte), T, replace=False)
lo, hi = Xc.min(0), Xc.max(0)
span = np.maximum(hi - lo, 1e-6)
is_int = np.all(np.abs(X - np.round(X)) < 1e-6, axis=0)  # integer / one-hot columns
mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
Zc = (Xc - mu) / sd
# typical nearest-neighbour distance between real rows (standardised space)
dd = np.sqrt(((Zc[:200, None] - Zc[None]) ** 2).sum(-1)); dd[np.arange(200), np.arange(200)] = np.inf
R_NN = float(np.median(dd.min(1)))

clf = core.make_diff_clf(n_estimators=1, seed=seed)
Xc_t = torch.tensor(Xc, device=dev)
yc_t = torch.tensor(yc, device=dev, dtype=torch.float32)
lo_t, span_t = torch.tensor(lo, device=dev), torch.tensor(span, device=dev)
mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)


def p_target(Xp, yp, xt):
    Xa = torch.cat([Xc_t, Xp]); ya = torch.cat([yc_t, yp])
    return core.proba(clf, Xa, ya, xt)[0]


def eval_rows(Xp_np, lab, xt):
    with torch.no_grad():
        Xp = torch.tensor(np.asarray(Xp_np, dtype=np.float32), device=dev)
        yp = torch.full((len(Xp_np),), float(lab), device=dev)
        return float(p_target(Xp, yp, xt)[lab])


def snap(Xp):
    Xp = np.clip(Xp, lo, hi)
    Xp[:, is_int] = np.round(Xp[:, is_int])
    return Xp


def plaus(Xp, xt_np):
    Z = (Xp - mu) / sd
    dnn = np.sqrt(((Z[:, None] - Zc[None]) ** 2).sum(-1)).min(1)
    dt = np.sqrt((((Xp - xt_np) / sd) ** 2).sum(-1))
    return float(np.mean(dnn) / R_NN), float(np.mean(dt) / R_NN)


def optimise(xt, xt_np, lab, k, init, far):
    """Adam on unconstrained logits; X = lo + span*sigmoid(W). If far, project
    rows to stay >= R_NN (std. space) away from the target point."""
    u = np.clip((init - lo) / span, 0.02, 0.98)
    W = torch.tensor(np.log(u / (1 - u)), device=dev, requires_grad=True)
    opt = torch.optim.Adam([W], lr=LR)
    yp = torch.full((k,), float(lab), device=dev)
    best = None
    fac = 1.0  # projection radius factor, grown if snapping pulls rows inside R_NN
    for s in range(STEPS):
        Xp = lo_t + span_t * torch.sigmoid(W)
        p = p_target(Xp, yp, xt)[lab]
        loss = -torch.log(p + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if far:
                Xp = lo_t + span_t * torch.sigmoid(W)
                z = (Xp - xt[0]) / sd_t
                nrm = z.norm(dim=1, keepdim=True)
                scale = torch.clamp(fac * R_NN / (nrm + 1e-9), min=1.0)
                Xp = xt[0] + z * scale * sd_t
                Xp = torch.minimum(torch.maximum(Xp, lo_t + 1e-4 * span_t), lo_t + span_t * (1 - 1e-4))
                uu = (Xp - lo_t) / span_t
                W.copy_(torch.log(uu / (1 - uu)))
        Xs = snap((lo_t + span_t * torch.sigmoid(W)).detach().cpu().numpy())
        # snapping can move rows back near the target; validity checked on snapped rows
        ok = (not far) or valid_far(Xs, xt_np)
        if not ok:
            fac *= 1.1
        ps = eval_rows(Xs, lab, xt)
        key = (ok, ps)
        if best is None or key > (best[3], best[0]):
            best = (ps, Xs, s + 1, ok)
        if ok and ps > 0.5:
            break
    return best


def valid_far(Xs, xt_np):
    return bool(np.sqrt((((Xs - xt_np) / sd) ** 2).sum(-1)).min() >= R_NN * 0.999)


out = open(f"results/c3_{name}_shard{shard}.jsonl", "w")
my = [t for j, t in enumerate(tsel) if j % n_shards == shard]
t_start = time.time()
for j, ti in enumerate(my):
    xt_np = Xte[ti:ti + 1]
    xt = torch.tensor(xt_np, device=dev)
    with torch.no_grad():
        p0 = core.proba(clf, Xc_t, yc_t, xt)[0, 1].item()
    c = int(p0 > 0.5); lab = 1 - c
    base = dict(t=int(ti), p1=p0, pred=c, y_true=int(yte[ti]))
    opp = Xc[yc == lab]
    znoise = rng.standard_normal((5, X.shape[1]))
    dirs = znoise / np.linalg.norm(znoise, axis=1, keepdims=True)
    for k in KS:
        recs = {}
        # baseline: k random real rows of the opposite class (duplicated)
        Xr = opp[rng.choice(len(opp), k, replace=False)]
        recs["real_opp_rows"] = (eval_rows(Xr, lab, xt), Xr)
        # baseline: k uniform random rows within range
        Xu = snap(lo + span * rng.random((k, X.shape[1])))
        recs["uniform_random"] = (eval_rows(Xu, lab, xt), Xu)
        # baseline: k exact copies of the target with opposite label
        Xcp = np.repeat(xt_np, k, 0)
        recs["target_copies"] = (eval_rows(Xcp, lab, xt), Xcp)
        # baseline: near-copies at distance R_NN (random directions), snapped
        Xn = snap(xt_np + dirs[:k] * R_NN * sd)
        if not valid_far(Xn, xt_np):  # re-scale until snapped rows respect radius
            for f in (1.25, 1.5, 2.0, 3.0):
                Xn = snap(xt_np + dirs[:k] * R_NN * f * sd)
                if valid_far(Xn, xt_np):
                    break
        recs["near_copies_R"] = (eval_rows(Xn, lab, xt), Xn)
        # gradient-optimised, init at near copies, constrained to be >= R_NN from target
        t0 = time.time()
        b = optimise(xt, xt_np, lab, k, xt_np + dirs[:k] * R_NN * sd, far=True)
        recs["grad_opt_far"] = (b[0], b[1]); steps_far, tf = b[2], time.time() - t0
        # gradient-optimised, init at random real opposite rows, range-only constraint
        t0 = time.time()
        b2 = optimise(xt, xt_np, lab, k, Xr, far=False)
        recs["grad_opt_from_real"] = (b2[0], b2[1]); steps_r, tr = b2[2], time.time() - t0
        for m, (pl, Xp) in recs.items():
            dnn, dt = plaus(Xp, xt_np[0])
            r = dict(base, k=k, method=m, p_flip=pl, flipped=bool(pl > 0.5),
                     dnn_rel=dnn, dtarget_rel=dt)
            if m == "grad_opt_far":
                r.update(steps=steps_far, secs=tf)
            if m in ("grad_opt_far", "near_copies_R"):
                r["valid_far"] = valid_far(Xp, xt_np)
            if m == "grad_opt_from_real":
                r.update(steps=steps_r, secs=tr)
            out.write(json.dumps(r) + "\n")
        out.flush()
    print(f"[{j+1}/{len(my)}] t={ti} p1={p0:.3f} elapsed={time.time()-t_start:.0f}s", flush=True)
print("R_NN", R_NN, "done")
