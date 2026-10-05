# Second confirmation (mixed data, longer optimisation) — pre-registration

Written: 2026-10-02 18:14 EDT, BEFORE any run on the seeds below.

## Motivation (disclosed)
Confirmation 1 (docs/CONFIRMATION_PREREG.md) did not support H1 on German credit or Taiwan
with 12 optimisation steps. An EXPLORATORY development run (results/exploratory_steps40,
seeds 101–103, 9 targets per dataset) suggested 40 steps may help. That run motivated this
test and is excluded from it.

## Design
Identical to confirmation 1 except: optimisation steps = 40 (the matched black-box searches
therefore also receive the correspondingly larger query budget); datasets = credit-g, taiwan
only; context seeds = 401, 402, 403, 404, 405, 406; targets = test-split indices 0..4 per seed.
60 audits. Implementation frozen at the source hashes recorded in
results/confirmation2/protocol.json at launch. stable_backward = ON.

## Primary endpoint and hypothesis (H2)
Same endpoint as confirmation 1: verified flip of standard TabPFN-3.5 with the exact appended
rows and labels, over all sampled targets. H2 (per dataset): gradient-search flip rate > the
better of the two query-matched black-box searches. Supported iff the lower bound of the
Bonferroni-adjusted (2 datasets) 97.5% two-level bootstrap interval is > 0. Both datasets
will be reported regardless of outcome. Confirmation 1 results stand as reported; this
test does not replace them.
