# Proofreading famous datasets — scan protocol (exploratory, fixed before running)

Written: 2026-10-04 10:41 EDT, before any run.

This is an exploratory scan, not a hypothesis test. Its protocol is fixed in advance so the gallery cannot be
cherry-picked after the fact.

## Datasets (fixed)
seaborn-data (github.com/mwaskom/seaborn-data, loaded by URL, not redistributed): penguins, tips, titanic, taxis,
planets, diamonds; scikit-learn California housing; UCI Iris (archive.ics.uci.edu/dataset/53, original `iris.data`).
Label column where a natural one exists: penguins species, titanic survived, planets method, iris species; none for
the others. Tables over 10,000 rows (diamonds, California housing, taxis is under) are scanned on a fixed random
10,000-row sample (seed 0) **plus** every row that has any value equal to 0 in a column whose minimum among non-zero
values is > 0 (so physically impossible zeros cannot be sampled away). Default threshold 2.

## Reporting rule
For each dataset, report the **top 10 flags** in rank order (all of them, not a selection). Each is labelled:
- **verified error**: impossible by definition or physics (e.g. zero length of a physical object, total ≠ sum of
  parts), or contradicted by an external source cited in the gallery;
- **plausible but unusual**: an external check or domain fact shows it can be real;
- **unverified**: not checked or not checkable.
Counts of each label per dataset are reported. No flag is described as an error without a citation or physical argument.

## Pre-specified known-answer check
UCI's `iris.names` documents two errors in `iris.data`: sample 35 (petal width 0.1, should be 0.2) and sample 38
(sepal width 3.1 should be 3.6, petal length 1.5 should be 1.4). We report the rank (or "not flagged") and the surprise
of each of these three cells, whatever the result.
