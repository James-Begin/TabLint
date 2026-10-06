"""Retained benchmark implementation; methodology and scope: docs/BENCHMARKS.md and docs/BENCHMARK_AMENDMENT.md."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

D = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/proofread_addendum')
by = defaultdict(list)
for p in sorted(D.glob('*_70*.json')):
    r = json.loads(p.read_text()); by[r['dataset']].append(r)
stored = {}
for p in Path('results/proofread').glob('*_70*.json'):
    r = json.loads(p.read_text()); stored[(r['dataset'], r['seed'])] = r['cells']['precision_at_k']['tabpfn_pit']
rows, drow = [], []
repro = []
for ds, rs in sorted(by.items()):
    m = lambda k: float(np.mean([r[k] for r in rs]))
    for r in rs:
        if (ds, r['seed']) in stored:
            repro.append(abs(stored[(ds, r['seed'])] - r['proofread_precision_at_k']))
    th = {t: {'precision': float(np.nanmean([r['threshold'][t]['precision'] if r['threshold'][t]['precision'] is not None else np.nan for r in rs])),
              'recall': float(np.mean([r['threshold'][t]['recall'] for r in rs])),
              'flagged_per_table': float(np.mean([r['threshold'][t]['flagged'] for r in rs]))} for t in ('2.0', '3.0')}
    rows.append({'dataset': ds, 'seeds': len(rs), 'proofread_p_at_k': m('proofread_precision_at_k'), 'ext_p_at_k': m('ext_precision_at_k'),
                 'proofread_row_auroc': m('proofread_row_auroc'), 'ext_row_auroc': m('ext_row_auroc'),
                 'proofread_cell_auroc': m('proofread_cell_auroc'), 'ext_cell_auroc': m('ext_cell_auroc'),
                 'ext_seconds': m('ext_seconds'), 'threshold': th})
    drow.append(m('proofread_row_auroc') - m('ext_row_auroc'))
d = np.array(drow)
pooled = {}
for t in ('2.0', '3.0'):
    fl = sum(r['threshold'][t]['flagged'] for rs in by.values() for r in rs)
    tf = sum(r['threshold'][t]['true_flagged'] for rs in by.values() for r in rs)
    er = sum(r['errors'] for rs in by.values() for r in rs)
    pooled[t] = {'precision': tf / fl if fl else None, 'recall': tf / er, 'flagged': fl, 'errors': er}
out = {'datasets': len(rows), 'runs': sum(len(v) for v in by.values()),
       'a1_row': {'wins': int((d > 0).sum()), 'mean_diff': float(d.mean()), 'wilcoxon_p_one_sided': float(wilcoxon(d, alternative='greater').pvalue) if len(d) > 5 else None},
       'a1_cell_wins': int(sum(r['proofread_p_at_k'] > r['ext_p_at_k'] for r in rows)),
       'determinism_max_abs_diff_vs_confirmation': float(max(repro)) if repro else None, 'a2_pooled': pooled, 'per_dataset': rows}
(D / 'summary.json').write_text(json.dumps(out, indent=2))
print(f"runs {out['runs']} datasets {out['datasets']}; reproduces confirmation P@k: max |diff| = {out['determinism_max_abs_diff_vs_confirmation']}")
print(f"A1 row-level: Proofread higher AUROC on {out['a1_row']['wins']}/{len(rows)}; mean diff {out['a1_row']['mean_diff']:+.3f}; p = {out['a1_row']['wilcoxon_p_one_sided']}")
print(f"A1 cell-level: Proofread higher precision@k on {out['a1_cell_wins']}/{len(rows)}")
for t, v in pooled.items():
    print(f"A2 threshold {t}: pooled precision {v['precision']:.3f}, recall {v['recall']:.3f} ({v['flagged']} flagged, {v['errors']} errors)")
for r in rows:
    print(f"  {r['dataset']:18s} P@k {r['proofread_p_at_k']:.2f} vs {r['ext_p_at_k']:.2f} | row AUROC {r['proofread_row_auroc']:.3f} vs {r['ext_row_auroc']:.3f} | th3 P {r['threshold']['3.0']['precision']:.2f} R {r['threshold']['3.0']['recall']:.2f} | th2 P {r['threshold']['2.0']['precision']:.2f} R {r['threshold']['2.0']['recall']:.2f}")
