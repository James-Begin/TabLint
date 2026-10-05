"""README product shot rendered from a stored Proofread report (no mock-up): flagged cells + suggestions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from proofread.core import Report

rep = Report.load('results/proofread_demos/breast_injected.json')
iss = rep.issues[rep.issues.kind == 'cell'].head(8)
cols = ['mean radius', 'mean perimeter', 'mean area', 'mean smoothness', 'worst texture', 'mean texture']
rows = [r for r in dict.fromkeys(iss.row.tolist()) if any((r, c) in {(x.row, x.column) for x in iss.itertuples()} for c in cols)][:3]
flag = {(r.row, r.column): r for r in iss.itertuples()}
fig, ax = plt.subplots(figsize=(11, 3.9)); ax.axis('off')
cell_text = [[f"{rep.data.at[i, c]:.4g}" for c in cols] for i in rows]
tab = ax.table(cellText=cell_text, rowLabels=[f'row {i}' for i in rows], colLabels=[c.replace(' ', '\n') for c in cols], loc='upper center', cellLoc='center',
               bbox=[0.07, 0.36, 0.9, 0.6])
tab.auto_set_font_size(False); tab.set_fontsize(11)
for (ri, ci), cell in tab.get_celld().items():
    cell.set_edgecolor('#dddddd')
    if ri == 0:
        cell.set_facecolor('#f3f3f3'); cell.set_text_props(weight='bold')
    elif ci >= 0 and (rows[ri - 1], cols[ci]) in flag:
        cell.set_facecolor('#ffd6d6'); cell.set_text_props(color='#a00000', weight='bold')
notes = []
for r in iss.itertuples():
    if r.row in rows and r.column in cols:
        notes.append(f"row {r.row} · {r.column} = {r.value:.4g}   →   TabPFN expects {r.suggested:.4g}  (80%: {r.low:.4g}–{r.high:.4g})   ·   {r.cause}")
ax.text(0.07, 0.3, '\n'.join(notes), transform=ax.transAxes, fontsize=10, va='top', family='monospace', color='#333')
ax.set_title('Proofread on a table with injected errors (breast-cancer demo; stored TabPFN-3.5 output)', fontsize=12, loc='left', x=0.07)
Path('docs/figures').mkdir(parents=True, exist_ok=True)
fig.savefig('docs/figures/proofread_product.png', dpi=160, bbox_inches='tight'); print('ok', rows)
