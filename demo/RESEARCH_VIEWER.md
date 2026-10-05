# Chain of Custody

A replay-first Streamlit showcase for measured TabPFN-3.5 robustness audits.

> TabPFN learns from rows; we differentiate through that learning to stress-test a prediction.

The viewer reads **local audit JSON only**. It does not import the audit engine, run inference, load model weights, use a GPU, accept uploads, or request credentials. There are no demo metrics, generated examples, or fallback measurements. If no reports exist, it explains how to supply them.

## Run

From the `forensics` project root, using an existing environment with Streamlit and pandas available:

```sh
uv run streamlit run demo/app.py
```

The default directory is `results/reports`, resolved relative to the project, not the launch directory. To explicitly select another local directory:

```sh
uv run streamlit run demo/app.py -- --reports-dir /absolute/path/to/reports
```

Or set an environment variable:

```sh
DECISION_STRESS_REPORTS_DIR=/absolute/path/to/reports uv run streamlit run demo/app.py
```

CLI selection takes precedence over the environment variable. Relative overrides resolve against the launch directory. Only direct `.json` files are listed; subdirectories and JSONL are not scanned. Symlinks escaping the selected directory are excluded. Use **Refresh reports** to reread newly written artifacts. No results are cached between reruns.

## Generate reports separately

Use the audit producer in its own inference environment, then export each completed audit to a JSON file in the selected directory. The viewer does not supply an inference command: execution, model access, and export are the producer's responsibility. Existing legacy results must be explicitly exported to the contract below; renaming a JSONL file is not conversion. Prefer writing to a temporary file and atomically replacing the destination after export completes.

### Artifact structure

Each file is a JSON object with `schema_version` exactly `"1.0"`. This is a field reference, **not a synthetic example report**:

| Field | Contract |
| --- | --- |
| `schema_version` | String `"1.0"` |
| `config` | Object containing the actual experiment configuration, decision rule/threshold and tie handling, preprocessing, constraints, seeds, and budget/search settings |
| `baseline` | Object: `model` = `"tabpfn_3_5_differentiable_surrogate"`, `p_positive` = positive-class probability, `predicted_class` = integer |
| `attacks` | List of attack objects, described below; an empty list is supported |
| `warnings` | List of producer warnings |
| `diagnostics` | Object of producer diagnostics |
| `feature_names` | Optional ordered list corresponding to raw feature positions |
| `target` | Optional raw feature list for the audited target, in the same order as attack rows |
| `context_preview` | Optional authorized, de-identified context preview |
| `dataset` / `dataset_metadata` | Optional dataset metadata, displayed as stored |
| `verification` | Optional list of independent model-verification objects |

Each attack object contains:

- `method`: method identifier; repeated methods/budgets remain separate entries.
- `n_rows`: reported row budget/count.
- `rows`: raw list of feature-value lists; `labels`: corresponding list of integer labels.
- `p_before`, `p_after`: measured positive-class probabilities, not probabilities of the attack label.
- `flipped`, `valid`: reported boolean flags. Omission or malformed values are **not** interpreted as false or passing.
- `constraints`: object containing `category_valid`, `in_bounds`, `protected_unchanged` boolean checks and `dtarget_min`, `dnn_max` recorded distance summaries. Document units, thresholds, applicability, and how validity was determined in `config`; the viewer does not evaluate constraints itself.
- `optimization_history`: optional list of objects. The viewer offers stored finite numeric fields as chart series; the horizontal axis is entry order, not an invented step counter.
- `evaluations`, `elapsed_s`, `masked_gradient_entries`: recorded search-effort diagnostics.

A verification object may contain `attack_method`, `model`, `p_before`, `p_after`, `flipped`, and `controls`, including `untouched_repeat`, `max_abs_repeat_delta`, and any other actual repeat controls. Preserve model identifiers, preprocessing/checkpoint provenance, and control definitions. The viewer displays verification separately from surrogate search and never manufactures deployed-model results. If several entries share a method, include row budget or another attack identifier in verification for unambiguous provenance; the viewer does not guess a linkage.

## What the interface shows

- Report picker and per-method measured before/after probability table and bar chart.
- Selected attack probability cards, with reported validity and decision-flip flags.
- Individual synthetic rows against the stored target, highlighting changed values. Raw feature representations are not decoded or assigned guessed units.
- Constraint checks, recorded distance summaries, search effort, and optional optimization trace.
- Separate model-transfer verification with recorded repeat controls.
- Producer warnings, experiment provenance, diagnostics, and optional context preview.
- JSON download of the loaded report. Non-finite numbers are normalized to `null`; the download is not a byte-identical copy of the original file.

There is **no editable threshold**. The viewer replays the producer's `flipped` flag and does not infer a decision from rounded probabilities or assume a decision rule. Missing/out-of-range probabilities are not plotted, baseline values are not silently substituted for missing attack values, and histories are not interpolated. A flip with invalid or unknown validity is not established as a feasible counterexample.

## Interpretation and safety

- **No counterexample found is not a robustness certificate.** A finite search only describes the evaluated settings, budgets, and methods; it does not establish an optimal attack budget.
- **Range and nearest-neighbor bounds are not domain realism.** Category validity, protected-feature checks, and distance bounds do not establish plausible people, records, or interventions.
- **No causal, fairness, or legal claim is made.** Feature comparison and protected-feature immutability are not fairness tests.
- Reference duplicate-row attacks are **unconstrained comparisons**, not evidence of feasible real-world changes. Review the producer's method and configuration before interpreting an attack.
- The differentiable surrogate is **not the deployed predictor**. Preprocessing, ensembling, model configuration, and repeat variability can affect transfer. Missing verification means transfer is unassessed, not successful; absent repeat controls mean repeatability is unassessed. No control tolerance is inferred.
- Use only authorized, de-identified reports. Local artifacts can contain sensitive raw rows, targets, and context; this app neither redacts data nor authenticates viewers. Do not put credentials into artifacts or expose the app publicly without access controls. Streamlit sends displayed values and downloadable JSON to connected browsers.
- Artifact values are rendered with standard Streamlit text/data/JSON components; no artifact-provided HTML is executed. Files are limited to 16 MiB. Unsupported schema versions, invalid JSON, and unreadable files produce actionable errors. Missing fields and malformed entries produce quality warnings or explicit unavailable states, not fabricated results. This is a bounded local replay interface, not a comprehensive schema validator or a hostile-input sandbox.

## Lightweight validation

Syntax validation without inference or importing the UI dependencies:

```sh
python -c 'import ast, pathlib; ast.parse(pathlib.Path("demo/app.py").read_text()); print("Syntax OK")'
```

`load_report(path)`, `summarize_report(report)`, `discover_reports(directory)`, and `report_directory(argv)` can be exercised independently: pandas and Streamlit are imported only inside `main()`. The app entry point is guarded by `if __name__ == "__main__"`.
