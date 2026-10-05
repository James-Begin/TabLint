# Numerical probe: evidence, not a validated robustness workaround

## Scope and budget

Local changes: `experiments/numerical_probe.py` and this document only. Remote execution: a GPU workstation (8× NVIDIA A10G), project directory, existing `uv run` environment, TabPFN 9.1.0. Only physical GPU7 was exposed; occupancy checked before each run. Two intra-op and two inter-op CPU threads; serial execution, no pool. Each foreground run used `timeout 900`.

Completed **20 target forwards, 1,444 total inference rows**, at most 101 rows per call (100 context/attack rows plus one target). Two initial startup failures occurred before any target forward. Model runs took 8.27 and 3.20 seconds. Additional operation-only runs made no target queries. GPU7 returned to 0 MiB after exit. Host free/uptime checked before/after: initial available RAM 736.7 GiB, final 736.3 GiB; uptime 19:24 → 19:32. No installations, authentication changes, production patches, commits, or pushes.

## Installed source evidence

Paths below are relative to `.venv/lib/python3.12/site-packages/tabpfn/`:

- `preprocessing/torch/ops.py:62–71`: square is evaluated on NaNs **before** `where` masks it; variance then enters `sqrt`.
- `preprocessing/torch/torch_standard_scaler.py:28–32`: zero std is replaced *after* sqrt. Constant-column microprobe produces `SqrtBackward0` NaNs.
- `preprocessing/torch/torch_soft_clip_outliers.py:63–72`: outliers become NaNs before second-pass `torch_nanstd`. A fixed 32-row singleton binary column with `n_sigma=4` reproduces **PowBackward0 at ops.py:65**, despite finite forward output. With default `n_sigma=12`, that tiny case is finite.
- `inference_config.py:370`: classification default threshold is 12. `preprocessing/torch/factory.py:141–148` adds clipping last.
- `architectures/tabpfn_v3_5.py:3275–3286`: ECDF division already guards zero-width intervals. Constant, binary, zero-heavy and unique ECDF microprobes had finite gradients.

## Checks and limitations

German-credit, raw Taiwan, breast cancer, and reduced Taiwan one-hot contexts all had finite original model gradients. Bit-identical row repeats had identical probabilities. Saved row JSON and SHA256 enable replay; the original Taiwan development context hash also matches exactly. Inspection of its original context plus near-row candidates finds **nine** columns masked as NaNs, consistent with its recorded **27** nonfinite entries across three poison rows. This supports the clipping/square mechanism but does **not** prove the original full-model failing backward: the larger context was inspected only, never rerun.

The isolated context-manager patch masks NaNs **before square** and uses exact sqrt forward with zero backward only at zero variance. It restores both imported function bindings. No `nan_to_num` gradient replacement. Tested forward differences were **0**, within explicit probability tolerance **1e-7**. Reduced Taiwan one-hot projected progress: 0.372783 → 0.467017, valid range/hard-one-hot candidate, no flip. The original gradient was already finite; this is not a demonstrated recovery.

Breast-cancer finite differences inside an empirical gap: autograd −0.00277869 versus central difference −0.00292425. ECDF tie test has zero gradients but rank jump 0.5 → 0.75 and central difference 1250. Local directions are not finite-change attribution.

Remote evidence: `results/numerical_probe/{summary,followup,ops_only}.json`, saved public-row JSON, and `logs/numerical_probe*.log`. **Keep audit defaults unchanged; workaround and full failing-case recovery remain unvalidated.**
