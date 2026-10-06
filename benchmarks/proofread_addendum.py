"""Retained benchmark implementation; methodology and scope: docs/BENCHMARKS.md and docs/BENCHMARK_AMENDMENT.md."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from benchmarks.common import load_any
from benchmarks.proofread_confirm import DATASETS, inject, tabpfn_reg


def tabpfn_clf(seed):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device='cuda:0', random_state=seed, ignore_pretraining_limits=True)


def ext_regressor(seed):
    """tabpfn-extensions 0.6.3 casts targets to the logits dtype (float32) while tabpfn 9.1.0 bar-distribution borders
    are float64, which raises in criterion.forward. Upcast logits to float64 (precision only; values unchanged)."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    class _Reg(TabPFNRegressor):
        def predict(self, X, *args, **kw):
            out = super().predict(X, *args, **kw)
            if kw.get('output_type') == 'full' or (args and args[0] == 'full'):
                out['logits'] = out['logits'].double()
            return out
    base = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device='cuda:0', random_state=seed, ignore_pretraining_limits=True)
    r = _Reg(); r.set_params(**base.get_params())
    return r


def proofread_scores(Xe, y, seed):
    n, d = Xe.shape; S = np.zeros((n, d))
    for j in range(d):
        Z = np.c_[np.delete(Xe, j, 1), y]; t = Xe[:, j]; pit = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Z):
            o = tabpfn_reg(seed).fit(Z[tr], t[tr]).predict(Z[te], output_type='full')
            lg = o['logits']; yy = torch.tensor(t[te], dtype=lg.dtype, device=lg.device)
            F = o['criterion'].cdf(lg, yy.unsqueeze(-1)).squeeze(-1)
            pit[te] = np.clip(np.nan_to_num(F.detach().double().cpu().numpy().reshape(-1), nan=.5), 1e-9, 1 - 1e-9)
        S[:, j] = -np.log(2 * np.minimum(pit, 1 - pit))   # natural log, as in the confirmation
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--seeds', default='701,702,703,704,705'); ap.add_argument('--out', default='results/proofread_addendum')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
    for ds in a.datasets.split(','):
        X0, y0 = load_any(ds); cont = [j for j in range(X0.shape[1]) if len(np.unique(X0[:, j])) > 10]
        for seed in map(int, a.seeds.split(',')):
            f = out / f'{ds}_{seed}.json'
            if f.exists():
                continue
            t0 = time.time(); rng = np.random.default_rng(seed)
            idx = rng.choice(len(y0), min(400, len(y0)), replace=False)
            X, y = X0[idx].astype(np.float64), y0[idx]
            if len(set(y)) < 2:
                continue
            X = X[:, cont]
            Xe, mask, kind = inject(X, .03, np.random.default_rng(seed))   # identical to proofread_confirm.cells
            n, d = Xe.shape; k = int(mask.sum()); m = mask.ravel(); row_err = mask.any(1)
            S = proofread_scores(Xe, y, seed)
            t1 = time.time()
            um = TabPFNUnsupervisedModel(tabpfn_clf=tabpfn_clf(seed), tabpfn_reg=ext_regressor(seed))
            um.fit(torch.tensor(Xe, dtype=torch.float32))
            dens = um.outliers(torch.tensor(Xe, dtype=torch.float32), n_permutations=10).detach().cpu().double().numpy()
            t_ext = time.time() - t1
            ext_row = np.nan_to_num(-dens, nan=0, posinf=1e12, neginf=-1e12)
            ext_cell = np.repeat(ext_row[:, None], d, 1).ravel()
            pr = np.nan_to_num(S.ravel(), nan=0, posinf=1e12)
            def pak(s):
                return float(m[np.argsort(-s, kind='stable')[:k]].mean())
            rec = {'dataset': ds, 'seed': seed, 'errors': k, 'rows_with_error': int(row_err.sum()), 'n': n, 'd': d,
                   'proofread_precision_at_k': pak(pr), 'ext_precision_at_k': pak(ext_cell),
                   'proofread_cell_auroc': float(roc_auc_score(m, pr)), 'ext_cell_auroc': float(roc_auc_score(m, ext_cell)),
                   'proofread_row_auroc': float(roc_auc_score(row_err, S.max(1))), 'ext_row_auroc': float(roc_auc_score(row_err, ext_row)),
                   'ext_seconds': t_ext, 'threshold': {}}
            log10 = S / np.log(10)
            for th in (2.0, 3.0):
                flag = log10.ravel() >= th
                rec['threshold'][str(th)] = {'flagged': int(flag.sum()), 'true_flagged': int((flag & m).sum()),
                                             'precision': float((flag & m).sum() / flag.sum()) if flag.sum() else None,
                                             'recall': float((flag & m).sum() / k)}
            rec['seconds'] = time.time() - t0
            f.write_text(json.dumps(rec, indent=2))
            print(ds, seed, 'P@k proofread %.2f ext %.2f | rowAUROC proofread %.3f ext %.3f | th3 P %s R %.2f' % (
                rec['proofread_precision_at_k'], rec['ext_precision_at_k'], rec['proofread_row_auroc'], rec['ext_row_auroc'],
                rec['threshold']['3.0']['precision'], rec['threshold']['3.0']['recall']), round(t_ext), flush=True)


if __name__ == '__main__':
    main()
