"""docs/figures/version_tradeoff.png from results/version_compare/summary.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = json.loads(Path('results/version_compare/summary.json').read_text())
name = {'V2': 'TabPFN v2', 'V3': 'TabPFN v3', 'V3_5': 'TabPFN-3.5', 'V3_5_FAST': 'TabPFN-3.5-Fast'}
fig, ax = plt.subplots(figsize=(7, 4.2))
for v, c in zip(name, ['#999999', '#777777', '#1f77b4', '#ff7f0e']):
    x, y = S['total_seconds'][v] / S['runs'], S['mean_p_at_k'][v]
    ax.scatter(x, y, s=160, color=c, zorder=3); ax.annotate(name[v], (x, y), xytext=(8, 6), textcoords='offset points', fontsize=10)
ax.set_xlabel('seconds per 400-row table (A10G GPU)'); ax.set_ylabel(f"mean precision@k ({len(S['per_dataset'])} datasets)")
values = list(S['mean_p_at_k'].values())
ax.set_ylim(min(values) - .01, max(values) + .015); ax.set_xlim(0, max(S['total_seconds'].values()) / S['runs'] * 1.3)
ax.set_title(f"TabLint by TabPFN version (amended scope, {S['runs']} tables)\n"
             f"3.5-Fast: precision difference vs 3.5 ({S['fast_vs_35']['mean_diff_p_at_k']:+.3f}), {S['fast_vs_35']['median_speedup']:.1f}× faster (median)", fontsize=10)
spread = max(S['mean_p_at_k'].values()) - min(S['mean_p_at_k'].values())
ax.text(0.01, 0.02, f'y-axis zoomed: all four versions lie within {spread:.3f} of each other', transform=ax.transAxes, fontsize=8, color='#666')
ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig('docs/figures/version_tradeoff.png', dpi=170); print('ok')
