"""EXPLORATORY: cell-level error detection ("spellcheck for tables") with TabPFN-3.5 (dev seeds).

Inject errors into ~2% of numeric cells of a clean n-row table, by type:
  decimal : value ×10 or ÷10 (classic data-entry slip)
  swap    : value copied from another random row in the same column (marginally plausible!)
  offset  : value + 3 column-SDs
Scores per cell (higher = more suspicious), all out-of-fold (5 folds, each column predicted from the
other columns + label):
  zscore     : |x - median| / MAD of the column (marginal, standard)
  iforest    : row-level IsolationForest score broadcast to cells (standard row outlier detector)
  rf_resid   : |x - RF prediction| / OOF residual SD of that column
  tabpfn_resid : same with TabPFN-3.5 regressor mean
  tabpfn_pit : two-sided tail surprise from TabPFN's full predictive distribution: -log(2 min(F, 1-F))
Metrics: cell AUROC overall and per error type; precision@#errors. Also correction quality: |median - truth|.
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from benchmarks.common import load_any


def tabpfn_reg(seed):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device='cuda:0', random_state=seed,
                                                      ignore_pretraining_limits=True)


def inject(X, rate, rng):
    Xe = X.copy(); n, d = X.shape
    mask = np.zeros((n, d), bool); kind = np.full((n, d), '', object)
    sd = X.std(0) + 1e-9
    cells = rng.choice(n * d, int(rate * n * d), replace=False)
    types = ['decimal', 'swap', 'offset']
    for t, c in enumerate(cells):
        i, j = divmod(int(c), d); k = types[t % 3]
        if k == 'decimal':
            new = X[i, j] * (10 if rng.random() < .5 else .1)
            if abs(new - X[i, j]) < 0.5 * sd[j]:
                new = X[i, j] + 3 * sd[j]; k = 'offset'
        elif k == 'swap':
            new = X[rng.integers(n), j]
            if abs(new - X[i, j]) < 0.5 * sd[j]:
                continue  # an un-detectable no-op swap is not an error
        else:
            new = X[i, j] + (3 if rng.random() < .5 else -3) * sd[j]
        Xe[i, j] = new; mask[i, j] = True; kind[i, j] = k
    return Xe, mask, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='eeg-eye-state,MagicTelescope,breast-cancer,phoneme,wilt,electricity,diabetes,banknote')
    ap.add_argument('--seeds', default='101'); ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--rate', type=float, default=0.02); ap.add_argument('--out', default='results/pilots/cells')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets.split(','):
        X0, y0 = load_any(ds)
        keep = [j for j in range(X0.shape[1]) if len(np.unique(X0[:, j])) > 10]  # continuous columns only
        X0 = X0[:, keep]
        for seed in map(int, a.seeds.split(',')):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y0), min(a.n, len(y0)), replace=False)
            X, y = X0[idx].astype(np.float64), y0[idx]
            Xe, mask, kind = inject(X, a.rate, rng)
            n, d = X.shape; t0 = time.time()
            S = {k: np.zeros((n, d)) for k in ('zscore', 'iforest', 'rf_resid', 'tabpfn_resid', 'tabpfn_pit')}
            med = np.median(Xe, 0); mad = np.median(np.abs(Xe - med), 0) * 1.4826 + 1e-9
            S['zscore'] = np.abs(Xe - med) / mad
            S['iforest'] = np.repeat(-IsolationForest(random_state=seed).fit(Xe).score_samples(Xe)[:, None], d, 1)
            corr = np.zeros((n, d))
            for j in range(d):
                Z = np.c_[np.delete(Xe, j, 1), y]
                rf_pred = np.zeros(n); tp_mean = np.zeros(n); tp_med = np.zeros(n); pit = np.zeros(n)
                for tr, te in KFold(5, shuffle=True, random_state=seed).split(Z):
                    rf_pred[te] = RandomForestRegressor(100, n_jobs=2, random_state=seed).fit(Z[tr], Xe[tr, j]).predict(Z[te])
                    out_full = tabpfn_reg(seed).fit(Z[tr], Xe[tr, j]).predict(Z[te], output_type='full')
                    tp_mean[te] = out_full['mean']; tp_med[te] = out_full['median']
                    crit = out_full['criterion']; logits = out_full['logits']
                    yy = torch.tensor(Xe[te, j], dtype=logits.dtype, device=logits.device)
                    F = crit.cdf(logits, yy.unsqueeze(-1)).squeeze(-1).detach().float().cpu().numpy() if crit.cdf(logits, yy.unsqueeze(-1)).dim() > 1 else crit.cdf(logits, yy).detach().float().cpu().numpy()
                    pit[te] = np.clip(np.nan_to_num(F.astype(np.float64), nan=0.5), 1e-9, 1 - 1e-9)
                for name, pred in (('rf_resid', rf_pred), ('tabpfn_resid', tp_mean)):
                    r = np.abs(Xe[:, j] - pred); S[name][:, j] = r / (np.median(r) * 1.4826 + 1e-9)
                S['tabpfn_pit'][:, j] = -np.log(2 * np.minimum(pit, 1 - pit))
                corr[:, j] = tp_med
            rec = {'dataset': ds, 'seed': seed, 'n': n, 'd': d, 'errors': int(mask.sum()), 'auroc': {}, 'precision_at_k': {}, 'by_type': {}}
            k = int(mask.sum())
            for name, s in S.items():
                f = np.nan_to_num(s.ravel(), nan=0.0, posinf=1e12); m = mask.ravel()
                rec['auroc'][name] = float(roc_auc_score(m, f))
                rec['precision_at_k'][name] = float(m[np.argsort(-f)[:k]].mean())
                for t in ('decimal', 'swap', 'offset'):
                    sel = (kind.ravel() == t) | ~m
                    if (kind.ravel() == t).sum():
                        rec['by_type'].setdefault(t, {})[name] = float(roc_auc_score(m[sel], f[sel]))
            # correction: relative error of TabPFN median vs truth, on error cells, compared with column median
            e = mask
            rec['correction'] = {'tabpfn_median_abs_err_sd': float((np.abs(corr[e] - X[e]) / (X.std(0)[np.where(e)[1]] + 1e-9)).mean()),
                                 'column_median_abs_err_sd': float((np.abs(med[np.where(e)[1]] - X[e]) / (X.std(0)[np.where(e)[1]] + 1e-9)).mean())}
            rec['seconds'] = time.time() - t0
            (out / f'{ds}_{seed}.json').write_text(json.dumps(rec, indent=2))
            print(ds, seed, 'd', d, 'err', k, 'AUROC', {m: round(v, 3) for m, v in rec['auroc'].items()},
                  'P@k', {m: round(v, 2) for m, v in rec['precision_at_k'].items()},
                  'swapAUROC', {m: round(v, 3) for m, v in rec['by_type'].get('swap', {}).items()}, rec['correction'], round(rec['seconds']), flush=True)


if __name__ == '__main__':
    main()
