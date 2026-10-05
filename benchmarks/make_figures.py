"""Figures from results/proofread/summary.json (pre-registered confirmation)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = json.loads(Path('results/proofread/summary.json').read_text())
out = Path('docs/figures'); out.mkdir(parents=True, exist_ok=True)
rows = sorted(S['cells_table'], key=lambda r: r['TabPFN precision@k'] - r['best baseline precision@k'])
fig, ax = plt.subplots(figsize=(8.6, 6.2))
for i, r in enumerate(rows):
    b, t = r['best baseline precision@k'], r['TabPFN precision@k']
    ax.plot([b, t], [i, i], color='#bbbbbb', lw=3, zorder=1)
    ax.scatter(b, i, color='#888888', s=70, zorder=2)
    ax.scatter(t, i, color='#1f77b4' if t > b else '#d62728', s=90, zorder=3)
    ax.text(min(b, t) - 0.015, i, f"{r['dataset']}", ha='right', va='center', fontsize=10)
    ax.text(max(b, t) + 0.015, i, f"vs {r['best baseline']}", ha='left', va='center', fontsize=8, color='#666')
ax.set_yticks([]); ax.set_xlim(0.05, 1.0); ax.set_xlabel('precision@k  (share of the top-k flagged cells that are real errors)')
h = S['h6_cells']
ax.set_title(f"Proofread (TabPFN-3.5) vs the best of 7 baselines per dataset\n"
             f"wins {h['wins']}/{h['datasets']} datasets · mean +{h['mean_diff']:.2f} · Wilcoxon p = {h['wilcoxon_p_one_sided']:.1g} (pre-registered)", fontsize=11)
ax.scatter([], [], color='#888888', label='best baseline (chosen after the fact)'); ax.scatter([], [], color='#1f77b4', label='TabPFN-3.5')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False); ax.spines[['top', 'right', 'left']].set_visible(False)
fig.tight_layout(); fig.savefig(out / 'proofread_hero.png', dpi=170)

types = S['by_error_type_auroc']; methods = ['zscore', 'iforest', 'ridge', 'rf', 'hgb_quantile', 'tabpfn_pit']
labels = {'zscore': 'z-score', 'iforest': 'IsolationForest', 'ridge': 'ridge', 'rf': 'random forest', 'hgb_quantile': 'HGB quantile', 'tabpfn_pit': 'TabPFN-3.5'}
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.13
for k, m in enumerate(methods):
    ax.bar([i + (k - 2.5) * w for i in range(len(types))], [t[m] for t in types], w, label=labels[m],
           color='#1f77b4' if m == 'tabpfn_pit' else plt.cm.Greys(0.3 + 0.1 * k))
ax.set_xticks(range(len(types))); ax.set_xticklabels([t['error type'] for t in types]); ax.set_ylim(0.5, 1.0)
ax.set_ylabel('cell AUROC'); ax.set_title('Detection by error type (14 datasets × 5 seeds)', pad=28); ax.legend(ncol=6, fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0))
ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(out / 'proofread_error_types.png', dpi=170)
print('ok')
