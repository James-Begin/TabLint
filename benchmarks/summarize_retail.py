"""Regenerate the local retail report and graph from completed raw records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LABELS = {'V2': 'V2', 'V3': 'V3', 'V3_5': '3.5 Base', 'V3_5_FAST': '3.5 Fast', 'sku_median_mad': 'SKU median/MAD'}
COLORS = {'V2': '#9b6baf', 'V3': '#d99434', 'V3_5': '#157f79', 'V3_5_FAST': '#3c78c0', 'sku_median_mad': '#7c8492'}


def summarize(root):
    records = [json.loads(p.read_text()) for p in sorted(root.glob('*.json'))]
    records = [r for r in records if 'model' in r]
    groups = {}
    for r in records:
        key = (r['train_rows'], r['model'], r.get('ablation', 'full'))
        groups.setdefault(key, []).append(r)
    rows = []
    for (size, model, ablation), runs in sorted(groups.items()):
        done = [r for r in runs if r['status'] == 'ok']
        row = {'train_rows': size, 'model': model, 'ablation': ablation,
               'completed': len(done), 'attempted': len(runs),
               'seeds_completed': [r['seed'] for r in done],
               'statuses': [r['status'] for r in runs]}
        if done:
            for field in ['precision_at_k', 'auroc', 'clean_log_mae', 'clean_price_mae',
                          'injected_error_log_mae_after', 'clean_80pct_interval_coverage']:
                vals = [r['metrics'][field] for r in done if field in r['metrics']]
                if vals:
                    row[field] = float(np.mean(vals))
                    row[field + '_min'] = float(np.min(vals))
                    row[field + '_max'] = float(np.max(vals))
            for field in ['fit_seconds', 'predict_seconds', 'wall_seconds', 'peak_rss_gib']:
                vals = [r[field] for r in done if field in r]
                if vals:
                    row[field] = float(np.mean(vals))
        rows.append(row)
    (root / 'summary.json').write_text(json.dumps({'scope': 'Local checkpoint capability experiment; not current TabLint integration', 'rows': rows}, indent=2) + '\n')
    keys = ['train_rows', 'model', 'ablation', 'completed', 'attempted', 'precision_at_k', 'auroc', 'clean_log_mae',
            'injected_error_log_mae_after', 'wall_seconds', 'peak_rss_gib']
    with (root / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return rows, records


def figure(rows, output, device="cpu"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'svg.hashsalt': 'tablint-retail'})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fields = [('clean_log_mae', 'Price prediction error', 'Mean absolute log error · lower is better'),
              ('precision_at_k', 'Injected error ranking', 'Precision among top 31 flags · higher is better'),
              ('wall_seconds', 'Local worker runtime', 'Seconds · completed runs only')]
    for ax, (metric, title, ylabel) in zip(axes, fields):
        for model, label in LABELS.items():
            group = sorted([r for r in rows if r['model'] == model and r['ablation'] == 'full' and metric in r], key=lambda r: r['train_rows'])
            if not group:
                continue
            x = [r['train_rows'] for r in group]
            y = [r[metric] for r in group]
            ax.plot(x, y, marker='o', color=COLORS[model], linewidth=2, label=label)
            # Seed ranges are descriptive; never present them as confidence intervals.
            if metric + '_min' in group[0]:
                ax.fill_between(x, [r[metric + '_min'] for r in group], [r[metric + '_max'] for r in group], color=COLORS[model], alpha=.12)
        ax.set_xscale('log')
        sizes = sorted(set(r['train_rows'] for r in rows))
        ax.set_xticks(sizes, [('>100k' if s == 100001 else f'{s / 1000:g}k') for s in sizes])
        ax.set_xlabel('Training rows')
        ax.set_title(title, fontweight='bold', loc='left', pad=14)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.15)
        if metric == 'wall_seconds':
            ax.set_yscale('log')
    axes[0].legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle(f'Retail descriptions and identifiers · local {device.upper()} experiment', x=.06, ha='left', fontsize=16, fontweight='bold')
    fig.text(.06, .015, 'One estimator · 1,024 future test rows · 3% synthetic price errors · shaded ranges show seeds, not confidence intervals.\nMissing points are unsupported or resource-limited attempts; see the report. Current-package text/date preprocessing is shared across checkpoints.', fontsize=8, color='#59616a')
    fig.tight_layout(rect=(.01, .12, 1, .92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=170, facecolor='white')
    fig.savefig(output.with_suffix('.svg'), facecolor='white', metadata={'Date': None})
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def report(rows, records, path, figure_path):
    device = ', '.join(sorted(set(r.get('device', 'cpu') for r in records if r['model'] != 'sku_median_mad')))
    lines = ['## Results', '',
             f'These are measurements using **{device}** and cached local checkpoints. They are a capability pilot, not evidence that the current TabLint UI accepts these context columns. Synthetic-error metrics concern only the injected errors; naturally unusual prices can also be flagged.', '',
             '| Context rows | Model | Completed / attempted | Precision@31 | AUROC | Clean log MAE ↓ | Worker seconds | Peak host RSS GiB |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if r['ablation'] != 'full':
            continue
        fmt = lambda k, decimals=3: f"{r[k]:.{decimals}f}" if k in r else '—'
        lines.append(f"| {r['train_rows']:,} | {LABELS[r['model']]} | {r['completed']}/{r['attempted']} | {fmt('precision_at_k')} | {fmt('auroc')} | {fmt('clean_log_mae')} | {fmt('wall_seconds', 1)} | {fmt('peak_rss_gib', 2)} |")
    failures = [r for r in records if r['status'] != 'ok']
    if failures:
        lines += ['', '**Incomplete attempts:**']
        for r in failures:
            reason = r.get('error', '')
            if len(reason) > 350:
                reason = reason[:350] + '…'
            lines += ['', f"- {LABELS[r['model']]} / {r['train_rows']:,} rows / seed {r['seed']} / {r.get('ablation', 'full')}: **{r['status']}**, {r.get('wall_seconds', 0):.1f}s, peak {r.get('peak_rss_gib', 0):.2f} GiB (when recorded). {reason}"]
    ablated = [r for r in rows if r['ablation'] != 'full']
    if ablated:
        lines += ['', '### Input ablations', '', '| Context rows | Model | Input | Seeds completed | Clean log MAE ↓ | Precision@31 |', '| ---: | --- | --- | ---: | ---: | ---: |']
        for r in ablated:
            if r['completed']:
                lines.append(f"| {r['train_rows']:,} | {LABELS[r['model']]} | {r['ablation']} | {r['completed']} | {r['clean_log_mae']:.3f} | {r['precision_at_k']:.3f} |")
    import os
    chart_url = os.path.relpath(figure_path.with_suffix('.png'), path.parent)
    lines += ['', f'![Local retail comparison]({chart_url})', '',
              'Values average completed seeds; a seed range is descriptive, not a confidence interval. Missing or timed-out models are not assigned scores. Paired quality claims require the same completed seeds at the same context size. Wall time includes worker startup, loading the prepared data and loading cached weights; downloads are excluded. Runtime of the SKU baseline is not directly comparable to the end-to-end model worker because it is measured in the parent process and omitted from that chart.', '',
              '[Raw measurements](../results/retail_cpu/) · [Machine-readable summary](../results/retail_cpu/summary.csv)', '']
    intro = path.read_text().split('## Results')[0] if path.exists() else '# Retail context experiment\n\n'
    text = intro + '\n'.join(line.rstrip() for line in lines)
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('results/retail_cpu'))
    ap.add_argument('--report', type=Path, default=Path('docs/RETAIL_CPU_BENCHMARK.md'))
    ap.add_argument('--figure', type=Path, default=Path('docs/figures/retail_cpu'))
    args = ap.parse_args()
    rows, records = summarize(args.root)
    device = ', '.join(sorted(set(r.get('device', 'cpu') for r in records if r['model'] != 'sku_median_mad')))
    figure(rows, args.figure, device)
    report(rows, records, args.report, args.figure)


if __name__ == '__main__':
    main()
