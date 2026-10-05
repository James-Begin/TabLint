"""Pre-registered confirmation (docs/PROOFREAD_PREREG.md): experiments C (cells) and L (labels)."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import IsolationForest, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from experiments.verify import factory
from benchmarks.common import load_any

DATASETS = ['eeg-eye-state', 'MagicTelescope', 'breast-cancer', 'phoneme', 'wilt', 'electricity', 'diabetes', 'banknote',
            'spambase', 'climate-crashes', 'kc1', 'steel-plates', 'ilpd', 'blood-transfusion']
BASE_C = ['zscore', 'iforest', 'ridge', 'knn', 'rf', 'hgb', 'hgb_quantile']


def tabpfn_reg(seed):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device='cuda:0', random_state=seed, ignore_pretraining_limits=True)


def lead_swap(v):
    s = f'{abs(v):.6e}'; m, e = s.split('e'); dg = m.replace('.', '')
    if len(dg) < 2 or dg[0] == dg[1]:
        return None
    dg = dg[1] + dg[0] + dg[2:]
    return float(('-' if v < 0 else '') + dg[0] + '.' + dg[1:] + 'e' + e)


def inject(X, rate, rng):
    Xe = X.copy(); n, d = X.shape; sd = X.std(0) + 1e-9
    mask = np.zeros((n, d), bool); kind = np.full((n, d), '', object)
    types = ['decimal', 'swap', 'offset', 'zero', 'transposition']
    for t, c in enumerate(rng.choice(n * d, int(rate * n * d), replace=False)):
        i, j = divmod(int(c), d); k = types[t % 5]; x = X[i, j]; new = None
        if k == 'transposition':
            new = lead_swap(x)
            if new is None or abs(new - x) < .5 * sd[j]:
                k = 'decimal'
        if k == 'zero':
            new = 0.0
            if abs(x) < .5 * sd[j]:
                k = 'offset'
        if k == 'decimal':
            new = x * (10 if rng.random() < .5 else .1)
            if abs(new - x) < .5 * sd[j]:
                k = 'offset'
        if k == 'swap':
            new = X[rng.integers(n), j]
            if abs(new - x) < .5 * sd[j]:
                continue
        if k == 'offset':
            new = x + (3 if rng.random() < .5 else -3) * sd[j]
        Xe[i, j] = new; mask[i, j] = True; kind[i, j] = k
    return Xe, mask, kind


def robust_scale(r):
    return r / (np.median(r) * 1.4826 + 1e-9)


def cells(X, y, seed, rate):
    rng = np.random.default_rng(seed)
    Xe, mask, kind = inject(X, rate, rng); n, d = X.shape
    S = {k: np.zeros((n, d)) for k in BASE_C + ['tabpfn_resid', 'tabpfn_pit']}
    med = np.median(Xe, 0); mad = np.median(np.abs(Xe - med), 0) * 1.4826 + 1e-9
    S['zscore'] = np.abs(Xe - med) / mad
    S['iforest'] = np.repeat(-IsolationForest(random_state=seed).fit(Xe).score_samples(Xe)[:, None], d, 1)
    corr = {k: np.zeros((n, d)) for k in ('tabpfn', 'rf', 'ridge')}
    for j in range(d):
        Z = np.c_[np.delete(Xe, j, 1), y]; t = Xe[:, j]
        P = {k: np.zeros(n) for k in ('ridge', 'knn', 'rf', 'hgb', 'q10', 'q50', 'q90', 'tp_mean', 'tp_med', 'pit')}
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Z):
            P['ridge'][te] = make_pipeline(StandardScaler(), RidgeCV()).fit(Z[tr], t[tr]).predict(Z[te])
            P['knn'][te] = make_pipeline(StandardScaler(), KNeighborsRegressor(10)).fit(Z[tr], t[tr]).predict(Z[te])
            P['rf'][te] = RandomForestRegressor(100, n_jobs=2, random_state=seed).fit(Z[tr], t[tr]).predict(Z[te])
            P['hgb'][te] = HistGradientBoostingRegressor(random_state=seed).fit(Z[tr], t[tr]).predict(Z[te])
            for q in (10, 50, 90):
                P[f'q{q}'][te] = HistGradientBoostingRegressor(loss='quantile', quantile=q / 100, random_state=seed).fit(Z[tr], t[tr]).predict(Z[te])
            o = tabpfn_reg(seed).fit(Z[tr], t[tr]).predict(Z[te], output_type='full')
            P['tp_mean'][te] = o['mean']; P['tp_med'][te] = o['median']
            lg = o['logits']; yy = torch.tensor(t[te], dtype=lg.dtype, device=lg.device)
            F = o['criterion'].cdf(lg, yy.unsqueeze(-1)).squeeze(-1)
            P['pit'][te] = np.clip(np.nan_to_num(F.detach().double().cpu().numpy().reshape(-1), nan=.5), 1e-9, 1 - 1e-9)
        for k in ('ridge', 'knn', 'rf', 'hgb'):
            S[k][:, j] = robust_scale(np.abs(t - P[k]))
        S['hgb_quantile'][:, j] = np.abs(t - P['q50']) / (np.maximum(P['q90'] - P['q10'], 0) + 1e-6 * (X[:, j].std() + 1e-9))
        S['tabpfn_resid'][:, j] = robust_scale(np.abs(t - P['tp_mean']))
        S['tabpfn_pit'][:, j] = -np.log(2 * np.minimum(P['pit'], 1 - P['pit']))
        corr['tabpfn'][:, j] = P['tp_med']; corr['rf'][:, j] = P['rf']; corr['ridge'][:, j] = P['ridge']
    k = int(mask.sum()); m = mask.ravel(); rec = {'errors': k, 'auroc': {}, 'precision_at_k': {}, 'by_type': {}}
    for name, s in S.items():
        f = np.nan_to_num(s.ravel(), nan=0, posinf=1e12)
        rec['auroc'][name] = float(roc_auc_score(m, f)); rec['precision_at_k'][name] = float(m[np.argsort(-f, kind='stable')[:k]].mean())
        for tname in ('decimal', 'swap', 'offset', 'zero', 'transposition'):
            pos = kind.ravel() == tname
            if pos.sum():
                rec['by_type'].setdefault(tname, {})[name] = float(roc_auc_score(m[pos | ~m], f[pos | ~m]))
    rec['types'] = {t: int((kind == t).sum()) for t in ('decimal', 'swap', 'offset', 'zero', 'transposition')}
    cols = np.where(mask)[1]; sdv = X.std(0)[cols] + 1e-9
    rec['correction_err_sd'] = {k: float((np.abs(v[mask] - X[mask]) / sdv).mean()) for k, v in corr.items()}
    rec['correction_err_sd']['column_median'] = float((np.abs(med[cols] - X[mask]) / sdv).mean())
    return rec


def labels(X, y, seed):
    rng = np.random.default_rng(seed); n = len(y)
    noisy = np.zeros(n, bool); noisy[rng.choice(n, int(.1 * n), replace=False)] = True
    yn = y.copy(); yn[noisy] = 1 - yn[noisy]
    out = {}
    for mname in ('logistic', 'knn', 'rf', 'hgb', 'tabpfn_standard'):
        s = np.zeros(n)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yn):
            mdl = make_pipeline(StandardScaler(), KNeighborsClassifier(10)) if mname == 'knn' else factory(mname, seed, 'cuda:0')
            p = mdl.fit(X[tr], yn[tr]).predict_proba(X[te]); s[te] = 1 - p[np.arange(len(te)), yn[te]]
        out[mname] = float(roc_auc_score(noisy, s))
    return {'auroc': out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASETS)); ap.add_argument('--seeds', default='701,702,703,704,705')
    ap.add_argument('--out', default='results/proofread')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets.split(','):
        X0, y0 = load_any(ds)
        cont = [j for j in range(X0.shape[1]) if len(np.unique(X0[:, j])) > 10]
        for seed in map(int, a.seeds.split(',')):
            f = out / f'{ds}_{seed}.json'
            if f.exists():
                continue
            t0 = time.time(); rng = np.random.default_rng(seed)
            idx = rng.choice(len(y0), min(400, len(y0)), replace=False)
            X, y = X0[idx].astype(np.float64), y0[idx]
            if len(set(y)) < 2:
                continue
            rec = {'dataset': ds, 'seed': seed, 'n': len(y), 'd_cont': len(cont), 'cells': cells(X[:, cont], y, seed, .03),
                   'labels': labels(X.astype(np.float32), y, seed)}
            rec['seconds'] = time.time() - t0
            f.write_text(json.dumps(rec, indent=2))
            c = rec['cells']['precision_at_k']
            print(ds, seed, 'P@k tabpfn %.2f best-base %.2f (%s)' % (c['tabpfn_pit'], max(c[b] for b in BASE_C), max(BASE_C, key=c.get)),
                  'labelAUROC', {k: round(v, 3) for k, v in rec['labels']['auroc'].items()}, round(rec['seconds']), flush=True)


if __name__ == '__main__':
    main()
