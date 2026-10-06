"""Generate numerical results from the retained runs and recalculated summaries."""
import json
from pathlib import Path
import numpy as np

S = json.loads(Path('results/proofread/summary.json').read_text())
count = S['h6_cells']['datasets']
L = ['# TabLint: numerical and label results', '',
     f"{count} retained public datasets × five seeds (701–705), 400-row tables, {S['n_runs']} runs. "
     'Two dataset executions were excluded after the author reported test errors. '
     '[Amendment and provenance](BENCHMARK_AMENDMENT.md) · [Scoring implementation](../benchmarks/proofread_confirm.py).', '',
     'The hypotheses and scoring rules were specified before running; the retained scope was amended after analysis. '
     'All retained datasets are reported. Statistics are conditional on that scope.', '',
     '## Headline', '', S['headline'].replace('  \n', '\n\n'), '',
     '![Numerical benchmark](figures/proofread_hero.png)', '',
     '## H6: cell errors (3% of cells corrupted; 5 error types)', '',
     'Precision@k is the share of the top k flagged cells that are injected errors, where k is the number injected. '
     'The comparator is the strongest measured method per dataset, chosen after averaging seeds: robust z-score, '
     'Isolation Forest, ridge, kNN, random forest, HGB or HGB quantiles. '
     'Fix error is mean absolute distance from the original clean value, in column standard deviations.', '',
     '| Dataset | TabPFN-3.5 | Best baseline | Baseline | Δ | TabPFN AUROC | Best baseline AUROC | Fix error TabPFN | RF | ridge | column median |',
     '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in S['cells_table']:
    L.append(f"| {r['dataset']} | {r['TabPFN precision@k']:.2f} | {r['best baseline']} | {r['best baseline precision@k']:.2f} | {r['difference']:+.2f} | "
             f"{r['TabPFN AUROC']:.3f} | {r['best baseline AUROC']:.3f} | {r['fix error (SD) TabPFN']:.2f} | {r['fix error RF']:.2f} | "
             f"{r['fix error ridge']:.2f} | {r['fix error column median']:.2f} |")
pit, residual = [], []
for row in S['cells_table']:
    runs = [json.loads(p.read_text()) for p in Path('results/proofread').glob(f"{row['dataset']}_70*.json")]
    pit.append(np.mean([r['cells']['precision_at_k']['tabpfn_pit'] for r in runs]))
    residual.append(np.mean([r['cells']['precision_at_k']['tabpfn_resid'] for r in runs]))
L += ['', f"**Distribution ablation:** predictive tail surprise beats the same TabPFN model’s point residual on "
      f"{sum(a > b for a, b in zip(pit, residual))}/{len(pit)} datasets; mean precision@k {np.mean(pit):.1%} vs {np.mean(residual):.1%}.", '',
      '![Detection by injected error type](figures/proofread_error_types.png)', '',
      '| Error type | ' + ' | '.join(k for k in S['by_error_type_auroc'][0] if k != 'error type') + ' |',
      '|---|' + '---:|' * (len(S['by_error_type_auroc'][0]) - 1)]
for row in S['by_error_type_auroc']:
    L.append(f"| {row['error type']} | " + ' | '.join(f"{v:.3f}" for k, v in row.items() if k != 'error type') + ' |')
L += ['', '## H7: label errors (10% of labels flipped uniformly)', '',
      '| Dataset | TabPFN-3.5 AUROC | Best baseline | Baseline AUROC | Δ |', '|---|---:|---|---:|---:|']
for r in S['labels_table']:
    L.append(f"| {r['dataset']} | {r['TabPFN AUROC']:.3f} | {r['best baseline']} | {r['best baseline AUROC']:.3f} | {r['difference']:+.3f} |")
L += ['', '## Baseline and threshold addendum', '', '[Addendum implementation](../benchmarks/proofread_addendum.py).']
A = json.loads(Path('results/proofread_addendum/summary.json').read_text())
L += ['', f"The same {A['runs']} corrupted tables were regenerated from their seeds. TabLint precision@k reproduces "
      f"the numerical confirmation exactly (maximum absolute difference {A['determinism_max_abs_diff_vs_confirmation']}).", '',
      "### A1. Prior Labs’ TabPFN outlier detector", '',
      f"Row AUROC: TabLint is higher on **{A['a1_row']['wins']}/{A['datasets']}** datasets, mean gain "
      f"{A['a1_row']['mean_diff']:+.3f}, one-sided Wilcoxon p = {A['a1_row']['wilcoxon_p_one_sided']:.3g}. "
      f"Cell precision@k: higher on **{A['a1_cell_wins']}/{A['datasets']}**.", '',
      '| Dataset | TabLint precision@k | Outlier detector precision@k | TabLint row AUROC | Outlier detector row AUROC | Detector seconds |',
      '|---|---:|---:|---:|---:|---:|']
for r in A['per_dataset']:
    L.append(f"| {r['dataset']} | {r['proofread_p_at_k']:.2f} | {r['ext_p_at_k']:.2f} | {r['proofread_row_auroc']:.3f} | {r['ext_row_auroc']:.3f} | {r['ext_seconds']:.0f} |")
L += ['', 'The optional `tabpfn-extensions` detector uses ten feature permutations and fits/scores the same table. '
      'Its row score is broadcast to cells. It required a float64 compatibility workaround on CUDA; '
      '[the bug report](bugs/TABPFN_EXTENSIONS_CUDA_DTYPE.md) records its effect and limitations. '
      'This compares the tested configurations on cell-error detection, not overall anomaly-detector quality.', '',
      '### A2. Operating point shown to users', '',
      f"| Threshold (−log₁₀ tail probability) | Precision | Recall | Cells listed ({A['runs']} tables) |", '|---|---:|---:|---:|']
for t, v in A['a2_pooled'].items():
    L.append(f"| {float(t):g} | {v['precision']:.1%} | {v['recall']:.1%} | {v['flagged']} |")
L += ['', 'Threshold 2 is the product default. This operating point differs from precision@k and was selected after the original threshold analysis.', '',
      '## Real-data context and limitations', '',
      'The stored [Pima report](../results/proofread_demos/pima_real.json) illustrates known impossible zeros and placeholder patterns. '
      'The [familiar-dataset gallery](FAMOUS_DATASETS.md) separates verified, unusual and unverified findings, including missed known errors.', '',
      '- Corruptions are synthetic; real-world error mixes and relationships differ.',
      '- These numerical experiments check continuous columns with more than ten distinct values. [Categorical checks](CATEGORICAL_RESULTS.md) are a separate experiment.',
      '- Each checked column costs five out-of-fold TabPFN fits; inference cost grows with table width.',
      '- Flags require source review. Predictive surprise is not a calibrated probability that a value is wrong.', '',
      'Generated from stored artifacts by `benchmarks/make_proofread_results.py`; no new inference is performed.']
Path('docs/PROOFREAD_RESULTS.md').write_text('\n'.join(L) + '\n')
print('Wrote numerical results for', count, 'datasets.')
