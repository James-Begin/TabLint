"""C4. Fragility audit across models, with categorical-aware optimisation.

For each target decision and budget k, build k synthetic training rows by several
attacks, append them to the training context with the label opposite to the model's
clean prediction, refit, and record whether the decision flips. The same rows are
evaluated on every model:

  tabpfn_diff  differentiable TabPFN-3.5 (n_estimators=1, identity preprocessing)
               - the model the optimiser sees
  tabpfn_std   standard TabPFN-3.5 (default sklearn path and settings) - deployed model
  hgb, rf, lr, knn  classical models refit on the poisoned data

Attacks (all rows labelled with the opposite class):
  target_copies   exact copies of the applicant
  near_copies     copies moved >= R_NN away from the applicant (random directions)
  real_opp        random real training rows of the opposite class (duplicated)
  grad_opt        gradient-optimised rows on tabpfn_diff, constrained to the observed
                  feature range, valid one-hot categories, and >= R_NN from the
                  applicant (R_NN = median nearest-neighbour distance between real rows)

Categorical features are optimised as softmax distributions over their categories
with an annealed temperature and evaluated after snapping to a single category.

Usage: python c4_audit.py <dataset> <n_ctx> <n_targets> <shard> <n_shards> [steps] [seed]
Writes results/c4_<dataset>_s<seed>_shard<i>.jsonl
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import sys
import time

import numpy as np
import torch

from fz import core, data

name = sys.argv[1]
N, T = int(sys.argv[2]), int(sys.argv[3])
shard, n_shards = int(sys.argv[4]), int(sys.argv[5])
STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 60
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 0
# variant "z": the differentiable surrogate sees standardised inputs (fixes non-finite
# / vanishing gradients on raw monetary features; the differentiable path has no
# input preprocessing of its own)
VARIANT = sys.argv[8] if len(sys.argv) > 8 else ""
LR = 0.1
KS = [1, 3, 5]
dev = "cuda"
rng = np.random.default_rng(SEED * 1000 + shard)

X, y, meta = data.load_meta(name, n_max=3000, seed=SEED)
groups, numeric = meta["groups"], meta["numeric"]
Xtr, Xte, ytr, yte = data.split(X, y, seed=SEED)
sel_rng = np.random.default_rng(SEED)
idx = sel_rng.choice(len(ytr), N, replace=False)
Xc, yc = Xtr[idx], ytr[idx]
tsel = sel_rng.choice(len(yte), T, replace=False)
lo, hi = Xc.min(0), Xc.max(0)
span = np.maximum(hi - lo, 1e-6)
is_int = np.zeros(X.shape[1], bool)
is_int[numeric] = np.all(np.abs(X[:, numeric] - np.round(X[:, numeric])) < 1e-6, axis=0)
mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
Zc = (Xc - mu) / sd
dd = np.sqrt(((Zc[:200, None] - Zc[None]) ** 2).sum(-1))
dd[np.arange(200), np.arange(200)] = np.inf
R_NN = float(np.median(dd.min(1)))

clf = core.make_diff_clf(n_estimators=1, seed=SEED)
Xc_t = torch.tensor(Xc, device=dev)
yc_t = torch.tensor(yc, device=dev, dtype=torch.float32)
lo_t, span_t, sd_t, mu_t = (torch.tensor(a, device=dev) for a in (lo, span, sd, mu))


def zs(t):
    """Input transform for the differentiable surrogate."""
    return (t - mu_t) / sd_t if VARIANT == "z" else t


Xc_s = zs(Xc_t)
num_t = torch.tensor(numeric, device=dev, dtype=torch.long)


# ------------------------------------------------------------------ models

def p_diff(Xp_np, lab, xt_np, cls=None):
    """Differentiable-path TabPFN: append rows Xp_np labelled `lab`; return P(cls)
    for the applicant (cls defaults to lab)."""
    cls = lab if cls is None else cls
    with torch.no_grad():
        Xp = torch.tensor(np.asarray(Xp_np, np.float32).reshape(-1, X.shape[1]), device=dev)
        yp = torch.full((len(Xp),), float(lab), device=dev)
        xt = torch.tensor(xt_np, device=dev)
        return float(core.proba(clf, torch.cat([Xc_s, zs(Xp)]), torch.cat([yc_t, yp]),
                                zs(xt))[0, cls])


def fit_predict(model, Xa, ya, xt_np):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model == "tabpfn_std":
        m = core.make_std_clf(n_estimators=8, seed=SEED)
    elif model == "hgb":
        m = HistGradientBoostingClassifier(random_state=SEED)
    elif model == "rf":
        m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=4)
    elif model == "lr":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    elif model == "knn":
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10))
    else:
        raise ValueError(model)
    m.fit(Xa, ya)
    return float(m.predict_proba(xt_np)[0, 1])


MODELS = ["tabpfn_std", "hgb", "rf", "lr", "knn"]


# ------------------------------------------------------------------ attacks

def snap(Xp):
    Xp = np.clip(Xp, lo, hi).astype(np.float32)
    Xp[:, is_int] = np.round(Xp[:, is_int])
    for g in groups:  # one category per group
        hot = np.argmax(Xp[:, g], axis=1)
        Xp[:, g] = 0.0
        Xp[np.arange(len(Xp)), g[hot]] = 1.0
    return Xp


def dist_to_target(Xp, xt_np):
    return np.sqrt((((Xp - xt_np) / sd) ** 2).sum(-1))


def valid_far(Xp, xt_np):
    return bool(dist_to_target(Xp, xt_np).min() >= R_NN * 0.999)


def plaus(Xp, xt_np):
    Z = (Xp - mu) / sd
    dnn = np.sqrt(((Z[:, None] - Zc[None]) ** 2).sum(-1)).min(1)
    return float(np.mean(dnn) / R_NN), float(np.mean(dist_to_target(Xp, xt_np)) / R_NN)


def near_copies(xt_np, k):
    for f in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        z = rng.standard_normal((k, X.shape[1]))
        z[:, [i for g in groups for i in g]] = 0.0  # keep applicant's categories...
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
        Xn = snap(xt_np + z * R_NN * f * sd)
        if valid_far(Xn, xt_np):
            return Xn
    # ...unless numeric moves alone cannot reach the radius: also resample categories
    Xn = snap(xt_np + rng.standard_normal((k, X.shape[1])) * R_NN * 2 * sd)
    return Xn


nonfinite = {"count": 0}
masked = {"entries": 0}


def grad_opt(xt_np, lab, k, init):
    """Numeric columns: X = lo + span*sigmoid(W). Each categorical group: softmax(L/tau)
    with tau annealed 1 -> 0.1. Penalty keeps rows >= R_NN from the applicant.
    Candidates are always judged after snapping to valid rows."""
    xt = torch.tensor(xt_np, device=dev)
    u = np.clip((init - lo) / span, 0.02, 0.98)
    W = torch.tensor(np.log(u / (1 - u))[:, numeric], device=dev, requires_grad=True)
    Ls = [torch.tensor(np.log(np.clip(init[:, g], 0.05, 1.0)) * 2.0, device=dev,
                       requires_grad=True) for g in groups]
    opt = torch.optim.Adam([W] + Ls, lr=LR)
    yp = torch.full((k,), float(lab), device=dev)
    best = None
    for s in range(STEPS):
        tau = max(0.1, 1.0 * (0.1 ** (s / max(1, STEPS - 1))))
        cols = torch.zeros((k, X.shape[1]), device=dev)
        cols[:, num_t] = lo_t[num_t] + span_t[num_t] * torch.sigmoid(W)
        for g, L in zip(groups, Ls):
            cols[:, torch.tensor(g, device=dev)] = torch.softmax(L / tau, dim=1)
        p = core.proba(clf, torch.cat([Xc_s, zs(cols)]), torch.cat([yc_t, yp]), zs(xt))[0, lab]
        d = (((cols - xt[0]) / sd_t) ** 2).sum(1).sqrt()
        loss = -torch.log(p + 1e-8) + 5.0 * torch.relu(1.05 * R_NN - d).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        if not torch.isfinite(loss):
            nonfinite["count"] += 1
            break  # keep the best valid candidate found so far
        for t in [W] + Ls:  # some heavy-tailed columns give non-finite gradient
            if t.grad is not None and not torch.isfinite(t.grad).all():  # entries: hold
                masked["entries"] += int((~torch.isfinite(t.grad)).sum())  # them fixed
                t.grad = torch.nan_to_num(t.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        cand = cols.detach().cpu().numpy()
        if not np.isfinite(cand).all():
            nonfinite["count"] += 1
            break
        Xs = snap(cand)
        ok = valid_far(Xs, xt_np)
        ps = p_diff(Xs, lab, xt_np)
        if best is None or (ok, ps) > (best[2], best[0]):
            best = (ps, Xs, ok, s + 1)
        if ok and ps > 0.5 and s >= 5:
            break
    if best is None:  # no finite step at all: fall back to the (valid) initial rows
        Xs = snap(init)
        best = (p_diff(Xs, lab, xt_np), Xs, valid_far(Xs, xt_np), 0)
    return best


# ------------------------------------------------------------------ main loop

out_path = f"results/c4{VARIANT}_{name}_s{SEED}_shard{shard}.jsonl"
out = open(out_path, "w")
my = [t for j, t in enumerate(tsel) if j % n_shards == shard]
clean = {}
t_start = time.time()
for j, ti in enumerate(my):
    xt_np = Xte[ti:ti + 1]
    p_clean = {"tabpfn_diff": p_diff(np.zeros((0, X.shape[1])), 1, xt_np, cls=1)}
    for m in MODELS:
        p_clean[m] = fit_predict(m, Xc, yc, xt_np)
    lab_d = 1 - int(p_clean["tabpfn_diff"] > 0.5)
    opp = Xc[yc == lab_d]
    for k in KS:
        rows = {"target_copies": np.repeat(xt_np, k, 0), "near_copies": near_copies(xt_np, k),
                "real_opp": opp[rng.choice(len(opp), k, replace=len(opp) < k)]}
        t0 = time.time()
        b = grad_opt(xt_np, lab_d, k, rows["near_copies"])
        rows["grad_opt"] = b[1]
        opt_secs, opt_steps, opt_valid = time.time() - t0, b[3], b[2]
        for attack, Xp in rows.items():
            dnn, dt = plaus(Xp, xt_np[0])
            rec = dict(t=int(ti), k=k, attack=attack, dnn_rel=dnn, dtarget_rel=dt,
                       valid_far=valid_far(Xp, xt_np), y_true=int(yte[ti]))
            if attack == "grad_opt":
                rec.update(secs=opt_secs, steps=opt_steps, opt_valid=opt_valid,
                           nonfinite_total=nonfinite["count"],
                           masked_grad_entries_total=masked["entries"])
            assert np.isfinite(Xp).all(), attack
            # every model: label = opposite of ITS OWN clean prediction for model-agnostic
            # attacks; grad_opt rows are only meaningful toward TabPFN's flip direction.
            for m in ["tabpfn_diff"] + MODELS:
                c = int(p_clean[m] > 0.5)
                lab = 1 - c
                if attack == "real_opp":
                    Xm = Xc[yc == lab][rng.choice((yc == lab).sum(), k,
                                                  replace=(yc == lab).sum() < k)]
                else:
                    Xm = Xp
                if attack == "grad_opt" and lab != lab_d:
                    rec[f"{m}_flip"] = None  # model disagrees with TabPFN on the decision
                    continue
                Xa = np.concatenate([Xc, Xm]).astype(np.float32)
                ya = np.concatenate([yc, np.full(k, lab)])
                p_new = (p_diff(Xm, lab, xt_np, cls=1) if m == "tabpfn_diff"
                         else fit_predict(m, Xa, ya, xt_np))
                rec[f"{m}_p_clean"] = p_clean[m]
                rec[f"{m}_p_new"] = p_new
                rec[f"{m}_flip"] = bool(int(p_new > 0.5) != c)
            out.write(json.dumps(rec) + "\n")
        out.flush()
    print(f"[{j + 1}/{len(my)}] t={ti} elapsed={time.time() - t_start:.0f}s", flush=True)
print("R_NN", R_NN, "done")
