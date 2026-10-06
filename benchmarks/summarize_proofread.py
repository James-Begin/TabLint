"""Retained benchmark implementation; methodology and scope: docs/BENCHMARKS.md and docs/BENCHMARK_AMENDMENT.md."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

D = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/proofread')
BASE_C = ['zscore', 'iforest', 'ridge', 'knn', 'rf', 'hgb', 'hgb_quantile']
BASE_L = ['logistic', 'knn', 'rf', 'hgb']
recs = [json.loads(p.read_text()) for p in sorted(D.glob('*_7*.json'))]
by = defaultdict(list)
for r in recs:
    by[r['dataset']].append(r)
rng = np.random.default_rng(0)
cells_rows, label_rows, dc, dl = [], [], [], []
for ds, rs in sorted(by.items()):
    pk = {m: float(np.mean([r['cells']['precision_at_k'][m] for r in rs])) for m in BASE_C + ['tabpfn_resid', 'tabpfn_pit']}
    au = {m: float(np.mean([r['cells']['auroc'][m] for r in rs])) for m in BASE_C + ['tabpfn_pit']}
    best = max(BASE_C, key=pk.get); dc.append(pk['tabpfn_pit'] - pk[best])
    corr = {k: float(np.mean([r['cells']['correction_err_sd'][k] for r in rs])) for k in ('tabpfn', 'rf', 'ridge', 'column_median')}
    cells_rows.append({'dataset': ds, 'seeds': len(rs), 'TabPFN precision@k': round(pk['tabpfn_pit'], 3), 'best baseline': best,
                       'best baseline precision@k': round(pk[best], 3), 'difference': round(pk['tabpfn_pit'] - pk[best], 3),
                       'TabPFN AUROC': round(au['tabpfn_pit'], 3), 'best baseline AUROC': round(max(au[b] for b in BASE_C), 3),
                       'fix error (SD) TabPFN': round(corr['tabpfn'], 2), 'fix error RF': round(corr['rf'], 2),
                       'fix error ridge': round(corr['ridge'], 2), 'fix error column median': round(corr['column_median'], 2)})
    la = {m: float(np.mean([r['labels']['auroc'][m] for r in rs])) for m in BASE_L + ['tabpfn_standard']}
    bl = max(BASE_L, key=la.get); dl.append(la['tabpfn_standard'] - la[bl])
    label_rows.append({'dataset': ds, 'TabPFN AUROC': round(la['tabpfn_standard'], 3), 'best baseline': bl,
                       'best baseline AUROC': round(la[bl], 3), 'difference': round(la['tabpfn_standard'] - la[bl], 3)})


def test(d):
    d = np.asarray(d); p = float(wilcoxon(d, alternative='greater').pvalue)
    boots = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    return {'datasets': len(d), 'wins': int((d > 0).sum()), 'mean_diff': float(d.mean()),
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], 'wilcoxon_p_one_sided': p,
            'supported': p < 0.05}


h6, h7 = test(dc), test(dl)
headline = (f"**Cells (H6):** TabPFN-3.5 beats the best of 7 baselines (chosen per dataset, after the fact) on "
            f"**{h6['wins']}/{h6['datasets']} datasets**; mean precision@k gain {h6['mean_diff']:+.3f} "
            f"[{h6['ci95'][0]:+.3f}, {h6['ci95'][1]:+.3f}], one-sided Wilcoxon p = {h6['wilcoxon_p_one_sided']:.2g} — "
            f"{'supported' if h6['supported'] else 'not supported'}.  \n"
            f"**Labels (H7):** TabPFN-3.5 beats the best of 4 baselines on **{h7['wins']}/{h7['datasets']}**; mean AUROC gain "
            f"{h7['mean_diff']:+.3f} [{h7['ci95'][0]:+.3f}, {h7['ci95'][1]:+.3f}], p = {h7['wilcoxon_p_one_sided']:.2g} — "
            f"{'supported' if h7['supported'] else 'not supported'}.")
# per-error-type AUROC pooled
types = defaultdict(lambda: defaultdict(list))
for r in recs:
    for t, v in r['cells']['by_type'].items():
        for m, a in v.items():
            types[t][m].append(a)
type_rows = [{'error type': t, **{m: round(float(np.mean(v[m])), 3) for m in BASE_C + ['tabpfn_pit']}} for t, v in types.items()]
out = {'h6_cells': h6, 'h7_labels': h7, 'headline': headline, 'cells_table': cells_rows, 'labels_table': label_rows,
       'by_error_type_auroc': type_rows, 'n_runs': len(recs)}
(D / 'summary.json').write_text(json.dumps(out, indent=2))
print(headline.replace('**', '').replace('  \n', '\n'))
print('\nper dataset (cells):')
for r in cells_rows:
    print(f"  {r['dataset']:18s} TabPFN {r['TabPFN precision@k']:.2f} vs {r['best baseline']:12s} {r['best baseline precision@k']:.2f}  fix err {r['fix error (SD) TabPFN']:.2f} vs RF {r['fix error RF']:.2f} / median {r['fix error column median']:.2f}")
print('\nper dataset (labels):')
for r in label_rows:
    print(f"  {r['dataset']:18s} TabPFN {r['TabPFN AUROC']:.3f} vs {r['best baseline']:8s} {r['best baseline AUROC']:.3f}")
print('\nAUROC by error type:')
for r in type_rows:
    print(' ', r)
