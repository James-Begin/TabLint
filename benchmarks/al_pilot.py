"""EXPLORATORY: look-ahead active learning (expected error reduction) with TabPFN-3.5 (dev seeds).

EER: for each candidate x in a random shortlist of the pool, and each label c weighted by p(c|x),
refit on L ∪ {(x,c)} (one forward pass) and measure expected 0-1 risk (1 - max p) on a pool subsample.
Pick the candidate with the lowest expected risk. Compared with random and margin (uncertainty) sampling
using TabPFN, and margin sampling with RF / logistic (their own learners). Metric: test accuracy curve.
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from experiments.verify import factory
from benchmarks.common import load_any


def run(strategy, learner, X, y, pool, test, init, budget, seed, shortlist=40, evalset=300):
    rng = np.random.default_rng(seed)
    L = list(init); P = [i for i in pool if i not in set(init)]
    curve = []
    def fit(idx, ys=None):
        return factory(learner, seed, 'cuda:0').fit(X[idx], y[idx] if ys is None else ys)
    for step in range(budget + 1):
        m = fit(L)
        curve.append(float((m.predict(X[test]) == y[test]).mean()))
        if step == budget:
            break
        if strategy == 'random' or len(set(y[L])) < 2:
            pick = P[int(rng.integers(len(P)))]
        else:
            pp = m.predict_proba(X[P])
            if strategy == 'margin':
                pick = P[int(np.argmin(np.abs(pp[:, 1] - pp[:, 0])))]
            else:  # eer
                cand = rng.choice(len(P), min(shortlist, len(P)), replace=False)
                ev = rng.choice(len(P), min(evalset, len(P)), replace=False)
                best, bestv = None, np.inf
                for ci in cand:
                    risk = 0.0
                    for c in (0, 1):
                        idx = L + [P[ci]]; ys = np.r_[y[L], c]
                        q = factory(learner, seed, 'cuda:0').fit(X[idx], ys).predict_proba(X[[P[e] for e in ev]])
                        risk += pp[ci, c] * float((1 - q.max(1)).mean())
                    if risk < bestv:
                        best, bestv = ci, risk
                pick = P[int(best)]
        L.append(pick); P.remove(pick)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='eeg-eye-state,MagicTelescope,phoneme,breast-cancer')
    ap.add_argument('--seeds', default='101'); ap.add_argument('--budget', type=int, default=40)
    ap.add_argument('--out', default='results/pilots/al')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets.split(','):
        X, y = load_any(ds)
        for seed in map(int, a.seeds.split(',')):
            pool, test = train_test_split(np.arange(len(y)), train_size=min(1000, len(y) // 2), stratify=y, random_state=seed)
            test = test[:1000]
            rng = np.random.default_rng(seed)
            init = [int(rng.choice(pool[y[pool] == c])) for c in (0, 1)] + [int(i) for i in rng.choice(pool, 4, replace=False)]
            init = list(dict.fromkeys(init))
            rec = {'dataset': ds, 'seed': seed, 'budget': a.budget, 'curves': {}, 'seconds': {}}
            for strat, learner in (('random', 'tabpfn_standard'), ('margin', 'tabpfn_standard'), ('eer', 'tabpfn_standard'),
                                   ('margin', 'rf'), ('margin', 'logistic')):
                t = time.time(); rec['curves'][f'{strat}_{learner}'] = run(strat, learner, X, y, list(pool), test, init, a.budget, seed)
                rec['seconds'][f'{strat}_{learner}'] = time.time() - t
                c = rec['curves'][f'{strat}_{learner}']
                print(ds, seed, strat, learner, 'acc@10 %.3f @20 %.3f @40 %.3f mean %.3f' % (c[10], c[20], c[-1], np.mean(c)), round(time.time() - t), flush=True)
            (out / f'{ds}_{seed}.json').write_text(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
