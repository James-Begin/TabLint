# Follow-up: original-context numerical recovery

This follow-up extends, and does not replace, NUMERICAL_PROBE.md. Audit defaults remain unchanged.

## Replay

`experiments/recover_failures.py` reconstructs four original development contexts from stored context IDs. It asserts an exact SHA256 match before inference, reuses the actual near-row candidates and labels, and compares unmodified vs scoped-patch forward probabilities and feature gradients. Only GPU7 was visible; all calls were serial with two CPU threads and a subprocess timeout. No gradient entries were silently masked.

## Findings

| Original artifact | Nonfinite gradient entries before patch | After patch | Absolute probability difference | Completed patched steps | Valid surrogate flip |
|---|---:|---:|---:|---:|---|
| credit-g_101_0 | 0 at initial candidate | 0 | 0 | 12 | yes |
| credit-g_102_0 | 3 | 0 | 0 | 12 | no |
| taiwan_101_0 | 39 | 0 | 0 | 12 | no |
| taiwan_102_0 | 27 | 0 | 0 | 12 | no |

The three initial-candidate failing gradients were genuinely reproduced and recovered. All four patched audits finished without numerical abort. This is demonstrated recovery of specific original cases, not a proof covering all possible contexts.

The patch changes backward handling of NaN-masked variance and zero variance while retaining tested forward values. It is isolated in a restoring context manager. All patched bindings were restored after each case. Runtime was approximately 16 seconds.

## Limits

Forward equality has been tested on these candidates and the prior microprobes, not globally proved. A zero derivative at zero variance is an explicit backward convention. ECDF rank crossings remain nondifferentiable. One surrogate flip is not a receiver-model flip; these patched candidates have not yet been included in the larger receiver verification benchmark. No default deployment or package site-files were modified.

Artifacts: `results/numerical_recovery/results.json`; earlier diagnostics: `results/numerical_probe/`. All results are development diagnostics, not confirmatory robustness estimates.
