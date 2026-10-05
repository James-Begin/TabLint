# Confirmation benchmark — pre-registration

Written: 2026-10-02 17:16 EDT, BEFORE inspecting results of
results/development_stable and before any confirmation run. Development artifacts
(seeds 101–103, 201–203) are excluded from confirmation.

## Frozen implementation
Code is frozen at the source SHA256 hashes recorded in results/confirmation/protocol.json
(written automatically by experiments/run_development.py at launch). Any change after
launch invalidates the run; changes require a new pre-registration.

Settings: context 200 rows sampled from the seeded 70% training split; targets = indices
0..4 of the seeded 30% test split (never selected by outcome); k = 3 appended rows with
the label opposite to the surrogate's clean prediction; 12 gradient steps; Adam lr 0.08 in
context-standardized units; stable_backward = ON; constraints: context feature bounds,
declared one-hot groups (German credit; Taiwan sex/education/marriage/payment status),
distance to target >= 0.5 x median context nearest-neighbour radius (R_NN), distance to
nearest context row <= 1.5 x R_NN. Baselines: random search and coordinate search with
the same initialization, constraints, labels and number of model queries.

## Design
Datasets: breast-cancer (UCI CC BY 4.0), credit-g (UCI CC BY 4.0), taiwan (UCI CC BY 4.0).
Context seeds: 301, 302, 303, 304, 305, 306. Targets per seed: 5. Total 90 audits.

## Primary endpoint and hypothesis
Verified decision flip on the standard TabPFN-3.5 receiver (default sklearn path,
fingerprint ON, 4 estimators), using the exact appended rows and labels, threshold 0.5,
counted over ALL sampled targets (invalid candidates and aborted optimisations count as
non-flips).

H1 (per dataset): gradient-search flip rate > flip rate of the better of the two matched
black-box searches (better chosen per dataset by its observed rate — a conservative
choice for the comparator). Paired by target. Two-level bootstrap (context seeds, then
targets; 4000 resamples). H1 is supported for a dataset if the lower bound of the
Bonferroni-adjusted 98.33% interval (3 datasets) is > 0. We will report all three
datasets regardless of outcome. (Amended 10 minutes after writing, before any
results were inspected: Holm replaced by the simpler, more conservative Bonferroni.)

## Secondary (descriptive, no confirmatory claims)
Surrogate flip rates; transfer to tabpfn_nofingerprint, hgb, hgb_regularized, rf,
rf_regularized, logistic; gradient abort counts; candidate validity; queries and wall time;
benign-label controls; untouched-refit repeatability.

## Not claimed
No robustness certificates; no minimum attack budgets; no domain realism of synthetic rows;
no comparison to optimised attacks against non-TabPFN receivers.
