# Chain of Custody — research protocol

Status: protocol for a new confirmatory benchmark. Existing C1–C4 outputs are exploratory pilots, not evidence for deployment or formal robustness guarantees.

## Research claim
TabPFN-3.5 supplies a differentiable learning algorithm. Optimizing appended context features can find decision-changing datasets. Counterexamples must be validated on the actual audited estimator; surrogate success is a separate endpoint.

This is not a claim that TabPFN is uniquely vulnerable, that discovered records are realistic, or that the smallest possible attack has been found.

## Threat model
White-box access to a fitted binary predictor's context, synthetic feature additions and corrupted labels within an explicit row budget. We do NOT modify real client data, authentication or production services. Datasets are public research examples; outcomes are not medical/credit advice.

Two tracks must be separated:
1. Corrupted labels: adversary may attach an opposite-class label; bounds govern FEATURES only.
2. Clean-label existing records: labels retained, row duplication/reweighting only. No claim about label poisoning from this track.

Reference exact target duplication is unconstrained and must never appear as an equally constrained baseline.

## Data & splitting
Fit scaling, imputation, ranges, categorical vocabulary, nearest-neighbor thresholds and tuning on context/development data only. Never infer integer schema from test rows. Split groups/duplicates before sampling; retain IDs and record dataset source, license, checksum, split hashes and selection seeds. All targets count, including failed optimization and model disagreement. Report conditional transfer and total target denominator separately.

Use development seeds for algorithm changes. Freeze implementation and hyperparameters before fresh confirmation seeds. Public dataset reuse makes results replication, not external validation.

## Constraints
Numeric bounds; valid one-hot groups; explicit integer steps; immutable columns unchanged from seed; optional maximum distance to a real CONTEXT record and minimum distance from target. Validate after hard projection. Multi-column semantic rules (e.g. valid payment codes, nonnegative counts) must be separately defined.

Range validity and nearest-neighbor distance do not establish joint plausibility. Call these 'schema-valid, bounded perturbations', not 'realistic records'. Store actual rows and label assignments for every result.

## Controls
- Refit untouched context twice with fixed seed; report numerical repeatability.
- Permute unchanged context as an invariance diagnostic.
- Append identical candidate rows with original/benign labels, separately.
- Fingerprint enabled/disabled must be an explicit ablation; default-configuration changes can alter all rows' fingerprints when context size changes.
- Classifier configuration, preprocessing and ensemble size must be recorded. The identity-preprocessing differentiable model is a SURROGATE, not the deployed estimator.
- Empty/no-valid-candidate runs and nonfinite gradients are failures, not certified robust decisions.

## Matched baselines
Gradient method vs random search vs derivative-free coordinate search with SAME initial candidate, SAME feasible domain and SAME target-query budget. Retain step-0 candidate as incumbent. Count every objective call, including discrete verification. Report equal wall-clock budget as secondary (backward is more expensive than forward).

Do not compare a many-step optimizer only against a single random draw. Verify transferred attacks on fixed receiver models. Transfer does not establish superiority to direct attacks against those receivers.

## Endpoints
Primary: VERIFIED decision flips at threshold 0.5 on the recorded receiver model, unconditional over all sampled targets.
Secondary: surrogate flips; change in target probability; constraints satisfied; finite-gradient fraction; clean-model accuracy/AUROC; clean confidence-stratified rate; evaluations and latency; query-matched search efficiency.

Report per seed / context split rather than averaging away heterogeneity. Resample context seeds (outer level) and targets (inner level) for uncertainty, not individual poison rows. Paired comparisons use the SAME eligible targets. Report model disagreement counts, excluded/failed targets, and all tested datasets. Use multiplicity control or clearly label exploratory ablations.

## Model modifications
Validate integer-label equivalence for soft-label relaxation and finite-difference gradient agreement. Do not patch global PyTorch behavior; isolate or scope adapters. Test nonfinite backward passes (heavy-tailed features) independently of scalar objective. Defaults stop on invalid gradients; masking must be disclosed as an experimental workaround.

Fine-tuning is not automatic. If used, optimize an explicit objective (e.g. regularize verified context sensitivity while retaining clean accuracy), split entire tasks/context seeds before selection, and compare with simple defenses (duplicate merging, context weighting). Freeze checkpoint choice before final attack evaluation. Never train on final evaluation attacks.

## Deliverables
Replayable reports + CLI; interactive visualization of actual counterexamples; evidence-backed audit narrative; reproducible scripts, dependency pins and limitations; no git pushes from this session. UI must function without GPU by reading stored reports. Live inference explicitly opts into GPU use.
