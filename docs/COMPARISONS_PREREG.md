# Model comparisons (descriptive, secondary) — pre-registration

Written: 2026-10-03 11:19 EDT, before either analysis below was run.
Data: the 90 reports of confirmation 1 (seeds 301–306; implementation frozen). No new attacks are generated for D.

## D. Detection head-to-head
Same augmented contexts (context + the gradient attack's exact rows/labels) as docs/RESULTS.md §3.
Detector = exact leave-one-out |change in P(target)| per context row, computed with each model:
TabPFN-3.5 (existing result), HGB, random forest (100 trees), logistic regression.
Metrics: AUROC (appended vs real rows; tied scores get average rank), appended rows in top-k, wall-clock
seconds per audit. Reported per dataset, descriptive, no threshold. Caveat fixed in advance: the appended
rows were optimised against TabPFN, so TabPFN-based detectors have a home advantage; and only audits
where the gradient rows are schema-valid are the primary subset (all 90 also reported).

## E. Receiver-targeted fragility
For each receiver (standard TabPFN-3.5, HGB, RF, logistic) and each report, run the two query-matched
black-box searches (random, coordinate) with THAT receiver as the oracle: same start rows (near_rows),
same labels, same constraints, same budget as the gradient run (its recorded evaluations, 24), plus one
evaluation of the start candidate. A flip means a valid candidate changes the receiver's OWN clean decision.
Reports: unconditional flip rate per receiver (incl. audits where the receiver's clean decision disagrees
with the surrogate's), and the subset where it agrees. Compared with the transfer flip rate of the
TabPFN-crafted gradient rows (existing verification). No significance test; bootstrap CIs over seeds then
targets are descriptive. This measures fragility to a weak, fixed-budget search, NOT worst-case robustness.
