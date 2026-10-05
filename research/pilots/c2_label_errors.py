"""C2. Label-error detection: flip 5% of training labels, rank rows by suspicion.

Scores (higher = more suspicious):
  tabpfn_oof     : 1 - P_TabPFN(given label) from 5-fold out-of-fold (standard model)
  hgb_oof        : same with HistGradientBoosting (default params)
  lr_oof         : same with standardised logistic regression
  tabpfn_labelgrad: soft-label gradient data valuation. For each fold, context =
                   train folds, loss = CE of held-out fold (given labels); score_i =
                   linearised decrease of held-out loss if row i's label is flipped.
  tabpfn_oof+grad: rank average of tabpfn_oof and tabpfn_labelgrad
Usage: python c2_label_errors.py <dataset> <seed>
"""
import json
import sys
import time

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fz import core, data

name, seed = sys.argv[1], int(sys.argv[2])
dev = "cuda"
rng = np.random.default_rng(seed)
X, y = data.load(name, n_max=2000, seed=seed)
Xtr, _, ytr, _ = data.split(X, y, seed=seed)
n = len(ytr)
flip_idx = rng.choice(n, int(round(0.05 * n)), replace=False)
yn = ytr.copy(); yn[flip_idx] = 1 - yn[flip_idx]
is_flip = np.zeros(n, bool); is_flip[flip_idx] = True
skf = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, yn))


def oof(make):
    p = np.zeros(n)
    for tr, va in skf:
        m = make(); m.fit(Xtr[tr], yn[tr]); p[va] = m.predict_proba(Xtr[va])[:, 1]
    return np.where(yn == 1, 1 - p, p)  # 1 - P(given label)


scores, times = {}, {}
t0 = time.time(); scores["tabpfn_oof"] = oof(lambda: core.make_std_clf(n_estimators=4, seed=seed)); times["tabpfn_oof"] = time.time() - t0
t0 = time.time(); scores["hgb_oof"] = oof(lambda: HistGradientBoostingClassifier(random_state=seed)); times["hgb_oof"] = time.time() - t0
t0 = time.time(); scores["lr_oof"] = oof(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))); times["lr_oof"] = time.time() - t0

# soft-label gradient data valuation (differentiable model, 1 estimator)
t0 = time.time()
clf = core.make_diff_clf(n_estimators=1, seed=seed)
gsum, gcnt = np.zeros(n), np.zeros(n)
Xt_all = torch.tensor(Xtr, device=dev)
yt_all = torch.tensor(yn, device=dev, dtype=torch.float32)
with core.soft_labels(True):
    for tr, va in skf:
        tr_t = torch.tensor(tr, device=dev); va_t = torch.tensor(va, device=dev)
        ya = yt_all[tr_t].clone().requires_grad_(True)
        p = core.proba(clf, Xt_all[tr_t], ya, Xt_all[va_t])
        loss = torch.nn.functional.nll_loss(torch.log(p + 1e-8), yt_all[va_t].long(), reduction="sum")
        loss.backward()
        g = ya.grad.double().cpu().numpy()
        # flipping label of row i changes y_i by (1-2y_i); predicted loss change g*(1-2y)
        gsum[tr] += -g * (1 - 2 * yn[tr]); gcnt[tr] += 1
scores["tabpfn_labelgrad"] = gsum / gcnt
times["tabpfn_labelgrad"] = time.time() - t0

# exact counterpart: actually flip each context row's label, measure held-out loss change
if len(sys.argv) > 3 and sys.argv[3] == "exact":
    t0 = time.time()
    esum = np.zeros(n)
    with torch.no_grad():
        for tr, va in skf:
            tr_t = torch.tensor(tr, device=dev); va_t = torch.tensor(va, device=dev)
            yv = yt_all[va_t].long(); yc0 = yt_all[tr_t]
            def L(yy):
                p = core.proba(clf, Xt_all[tr_t], yy, Xt_all[va_t]).detach()
                return float(torch.nn.functional.nll_loss(torch.log(p + 1e-8), yv, reduction="sum"))
            L0 = L(yc0)
            for j, i in enumerate(tr):
                yy = yc0.clone(); yy[j] = 1 - yy[j]
                esum[i] += L0 - L(yy)  # positive = flipping row i helps held-out loss
    scores["tabpfn_exactflip"] = esum / gcnt
    times["tabpfn_exactflip"] = time.time() - t0
scores["tabpfn_oof+grad"] = rankdata(scores["tabpfn_oof"]) + rankdata(scores["tabpfn_labelgrad"])
scores["tabpfn+hgb_oof"] = rankdata(scores["tabpfn_oof"]) + rankdata(scores["hgb_oof"])

k = int(is_flip.sum())
res = {"dataset": name, "seed": seed, "n_train": n, "n_flipped": k}
for m, s in scores.items():
    top = np.argsort(-s)[:k]
    res[m] = {"auroc": float(roc_auc_score(is_flip, s)), "prec@k": float(is_flip[top].mean()),
              "secs": times.get(m)}
    print(m, res[m])
json.dump(res, open(f"results/c2_{name}_s{seed}.json", "w"), indent=1)
