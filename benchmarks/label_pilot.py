"""EXPLORATORY pilot: label-error detection & repair (development seeds only).

Inject uniform label flips into a training context; score each row's suspicion with:
  oof_<model>  : 1 - P(given label) from 5-fold out-of-fold predictions (Cleanlab-style self-confidence)
  grad_tabpfn  : first-order effect of flipping row i's label on TabPFN cross-validated NLL,
                 via d(CV NLL)/d(soft label_i) through TabPFN's in-context learning (unique to ICL)
Metrics: AUROC for flipped rows; and test accuracy of TabPFN after removing each method's top-q rows.
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from auditkit.cli import dataset
from experiments.verify import factory
from fz import core


def oof(model, X, y, seed):
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p = factory(model, seed, 'cuda:0').fit(X[tr], y[tr]).predict_proba(X[te])
        s[te] = 1 - p[np.arange(len(te)), y[te]]
    return s


def grad_score(X, y, seed, folds=5):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32, device='cuda:0')
    clf = core.make_diff_clf(seed=seed, device='cuda:0')
    yl = torch.tensor(y, dtype=torch.float32, device='cuda:0', requires_grad=True)
    total = torch.zeros(len(y), device='cuda:0')
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        tr_t = torch.tensor(tr, device='cuda:0'); te_t = torch.tensor(te, device='cuda:0')
        with core.soft_labels(True):
            p = core.proba(clf, Z[tr_t], yl[tr_t], Z[te_t])
        nll = -torch.log(p[torch.arange(len(te)), torch.tensor(y[te], device='cuda:0')].clamp_min(1e-6)).sum()
        g, = torch.autograd.grad(nll, yl)
        total += g.detach()
    g = total.cpu().numpy()
    # flipping label i changes y_i by (1-2y_i); predicted loss change = g_i*(1-2y_i); suspicion = reduction
    return -g * (1 - 2 * y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='breast-cancer,eeg-eye-state,MagicTelescope,phoneme,credit-g,spambase,diabetes')
    ap.add_argument('--seeds', default='101,102,103'); ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--noise', type=float, default=0.1); ap.add_argument('--out', default='results/pilots/label_pilot')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets.split(','):
        X, y, _ = dataset(ds) if ds in ('breast-cancer', 'credit-g', 'taiwan') else __import__('experiments.screen_regimes', fromlist=['load']).load(ds)
        X = X.astype(np.float32)
        for seed in map(int, a.seeds.split(',')):
            f = out / f'{ds}_{seed}_{a.noise}.json'
            if f.exists():
                continue
            t0 = time.time()
            ic, it = train_test_split(np.arange(len(y)), train_size=a.n, stratify=y, random_state=seed)
            it = it[:1000]
            rng = np.random.default_rng(seed)
            noisy = np.zeros(a.n, bool); noisy[rng.choice(a.n, int(a.noise * a.n), replace=False)] = True
            Xc, yc = X[ic], y[ic].copy(); yc[noisy] = 1 - yc[noisy]
            scores = {m: oof(m, Xc, yc, seed) for m in ('logistic', 'hgb', 'rf', 'tabpfn_standard')}
            scores['grad_tabpfn'] = grad_score(Xc, yc, seed)
            # rank-average combination of TabPFN OOF + gradient
            from scipy.stats import rankdata
            scores['tabpfn_oof+grad'] = rankdata(scores['tabpfn_standard']) + rankdata(scores['grad_tabpfn'])
            rec = {'dataset': ds, 'seed': seed, 'n': a.n, 'noise': a.noise, 'auroc': {}, 'acc_after_removal': {}}
            def acc(keep):
                m = factory('tabpfn_standard', seed, 'cuda:0').fit(Xc[keep], yc[keep])
                return float((m.predict(X[it]) == y[it]).mean())
            q = int(noisy.sum())
            rec['acc_noisy'] = acc(np.ones(a.n, bool)); rec['acc_oracle'] = acc(~noisy)
            for m, s in scores.items():
                rec['auroc'][m] = float(roc_auc_score(noisy, s))
                keep = np.ones(a.n, bool); keep[np.argsort(-s)[:q]] = False
                rec['acc_after_removal'][m] = acc(keep)
            rec['seconds'] = time.time() - t0
            f.write_text(json.dumps(rec, indent=2))
            print(ds, seed, {m: round(v, 3) for m, v in rec['auroc'].items()}, 'acc noisy %.3f oracle %.3f' % (rec['acc_noisy'], rec['acc_oracle']),
                  {m: round(v, 3) for m, v in rec['acc_after_removal'].items()}, flush=True)


if __name__ == '__main__':
    main()
