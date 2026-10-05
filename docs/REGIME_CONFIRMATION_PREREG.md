# Confirmation in nonlinear regimes — pre-registration

Written: 2026-10-03 14:35 EDT, before any audit on these datasets.

## Motivation (disclosed)
The regime screen (docs/REGIME_SCREEN_PREREG.md, results/regime_screen/table.md) selected tasks where
TabPFN-3.5 clearly beats logistic regression at context size 200. The datasets below are the real
(non-synthetic) qualifiers with at most 14 features: eeg-eye-state, mozilla4, phoneme, MagicTelescope.
(madelon also qualified but has 500 features and is excluded on cost; synthetic tasks are excluded as
non-public-data showcases.) Selection was on predictive gap only; no audit was run on these tasks before
this registration. The regime selection is a disclosed, deliberate choice of where to test, not a claim that
the results hold on tabular data in general.

## Design
Identical to confirmation 1 (k=3, 12 steps, lr 0.08, stable_backward ON, context 200, same constraints)
except: datasets above; context seeds 501–506; 5 targets per seed => 120 audits. Implementation frozen
at the source hashes in results/regimes/protocol.json at launch. Only numeric features (categorical columns
dropped; none exist in these four).

## Hypotheses (each per dataset; Bonferroni over 4 datasets => 98.75% two-level bootstrap intervals;
supported iff the lower bound > 0)
- H4 (primary): verified flip rate of standard TabPFN-3.5 for gradient search exceeds that of the better of
  random and coordinate search (paired by target, all targets counted).
- H5: exact leave-one-out AUROC for identifying the appended rows is higher using TabPFN than using
  logistic regression (paired by audit).

## Descriptive secondary analyses (no tests)
Detection by HGB/RF LOO, top-k recall, seconds; receiver-targeted fragility (24-query search per receiver),
transfer of gradient rows. Negative or null outcomes on any dataset will be reported in RESULTS.md.
