"""Aggregate detection results: mean AUROC and top-k hits per detector, with a
two-level (context seed, target) bootstrap 95% CI. Descriptive/secondary only."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DET = ['loo_influence', 'label_suspicion', 'outlier']
d = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/confirmation/detection')
recs = [json.loads(p.read_text()) for p in sorted(d.glob('*.json')) if p.name != 'summary.json']
rng = np.random.default_rng(0)


def ci(groups, n=4000):
    keys = list(groups)
    vals = []
    for _ in range(n):
        ks = rng.choice(keys, len(keys))
        v = np.concatenate([np.asarray(groups[k])[rng.integers(0, len(groups[k]), len(groups[k]))]
                            for k in ks])
        vals.append(v.mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


out = {}
for ds in sorted({r['dataset'] for r in recs}):
    for subset in ('all', 'standard_flip'):
        rs = [r for r in recs if r['dataset'] == ds and (subset == 'all' or r['standard_flip'])]
        if not rs:
            continue
        row = {'n': len(rs)}
        for det in DET:
            g = defaultdict(list)
            for r in rs:
                g[r['seed']].append(r[f'{det}_auroc'])
            m = float(np.mean([r[f'{det}_auroc'] for r in rs]))
            lo, hi = ci(g)
            hits = float(np.mean([r[f'{det}_hits_in_topk'] / r['k'] for r in rs]))
            all3 = float(np.mean([r[f'{det}_hits_in_topk'] == r['k'] for r in rs]))
            row[det] = {'auroc': m, 'auroc_ci95': [lo, hi], 'topk_recall': hits, 'all_k_found': all3}
        out[f'{ds}|{subset}'] = row
        print(f"{ds:14s} {subset:14s} n={len(rs):2d} " + ' | '.join(
            f"{det}: AUROC {row[det]['auroc']:.3f} [{row[det]['auroc_ci95'][0]:.3f},"
            f"{row[det]['auroc_ci95'][1]:.3f}] top-k recall {row[det]['topk_recall']:.2f}"
            for det in DET))
(d / 'summary.json').write_text(json.dumps(out, indent=2))
