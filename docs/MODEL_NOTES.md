# Model and numerical notes

The live model is explicitly TabPFN-3.5 via tabpfn 9.1.0. Gradients are with respect to appended context FEATURES; attack labels are fixed binary values. We do not require the pilot's soft-label interpolation patch for this search.

## Forward versus backward
The adapter freezes pretrained model parameters and only requests input gradients, preventing model-gradient accumulation. A fixed context-only standardization feeds the differentiable path. Receiver verification uses separately named standard estimators; identity preprocessing is not silently substituted for deployment preprocessing.

TabPFN-3.5 also encodes empirical CDF ranks. Its rank channel is not globally smooth: exact duplicates and rank crossings can make central finite differences disagree with local autograd derivatives. Gradient checks must report the perturbation, whether a rank tie/crossing was encountered, and forward repeatability. Do not call a search direction an exact finite-change influence function.

A local finite-difference diagnostic found close agreement for some numeric coordinates and substantial disagreement for others; this is not a global gradient certification. Optimizer proposals are always hard-projected and forward-verified. Mixed/discrete and heavy-tailed data can give nonfinite gradients; the production audit engine stops and retains its incumbent instead of silently masking those derivatives.

## Experimental paths not enabled by default
- Soft-label interpolation from fz/core.py: integer-label equivalence and finite-difference behavior have pilot checks; global hooks make it unsuitable for concurrent threaded inference. Live MCP inference is serialized. The default audit never enables soft-label mode.
- Straight-through or smoothed ECDF gradients: interesting model modification, but must preserve/check forward outputs and be separately named. Not implemented yet.
- Fine-tuned robustness: not enabled. Must use held-out tasks/context seeds, clean-performance constraints, unseen attacks, and simple-defense comparators before any claim of improvement.

## Failure semantics
A finite valid incumbent can exist even if gradient optimization aborts. Display 'optimizer aborted; incumbent retained' and the abort reason. A baseline candidate flip is not evidence that gradients helped. Never pool fallback successes as successful optimization steps without this stratification.
