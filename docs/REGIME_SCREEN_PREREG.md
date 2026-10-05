# Regime screen: where does TabPFN-3.5 beat logistic regression? — pre-registration

Written: 2026-10-03 14:30 EDT, before running the screen.

## Purpose and disclosure
Earlier results show TabPFN's advantage on the audit tasks is modest (leave-one-out detection ties
logistic regression on the three original datasets). This screen looks for binary tabular tasks where
TabPFN-3.5 is *clearly* better than logistic regression at the audit's context size, so the audit
comparisons can be repeated where the gap exists. This is deliberate regime selection, and will be
described as such. Every screened candidate is reported, including those with no gap.

## Candidates (fixed before running)
OpenML 1489 phoneme, 1471 eeg-eye-state, 1120 MagicTelescope, 1462 banknote-authentication, 44 spambase,
40983 wilt, 1487 ozone-level-8hr, 1067 kc1, 1464 blood-transfusion, 1467 climate-model-simulation-crashes,
1480 ilpd, 37 diabetes, 1504 steel-plates-fault, 151 electricity, 1068 pc1, 1053 jm1, 40900 Satellite,
1046 mozilla4, 1485 madelon; plus two synthetic nonlinear tasks (xor-with-noise, two-moons-with-noise).
Rules: numeric features only (categorical columns dropped and the count disclosed); binary target
(majority/minority as labelled); up to 3000 rows sampled with a fixed seed.

## Protocol
Development seeds 101–110 (disjoint from confirmation seeds 301–306 and 401–406). Per seed: stratified
200-row context, up to 1000 held-out test rows. Models: logistic regression (standardized), HGB, random
forest, standard TabPFN-3.5. Metrics: accuracy and AUROC. Report the mean paired difference
(TabPFN − logistic) with a bootstrap CI over seeds.

## Selection rule
A dataset qualifies as a "nonlinear regime" iff the lower 95% bound of the paired accuracy difference
(TabPFN − logistic) is ≥ +0.05. Qualifying datasets go to a separate, pre-registered audit confirmation
on fresh seeds. All candidates and their gaps are published whatever the outcome.
