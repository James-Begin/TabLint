# TabLint v9 visual review — revision 15

Reviewed the encoded 1920×1080, 30 fps MP4, rather than relying only on source previews. Runtime: 87.7667 seconds; frames: 0–2632.

## Correction

The opening field-highlight scene froze source time at 0.51 seconds while the camera was still easing toward its final pose at 0.65 seconds. Resuming source time at 2.6 seconds produced a small position and approximately 1% scale snap after the horsepower labels. The hold now starts at 0.65 seconds, after the camera settles. Its duration was adjusted to preserve the export's total frame count and later cue positions. The enlarged 28 px labels remain unchanged.

## Visual coverage

All 2,633 consecutive encoded frames were visually inspected in 88 chronological contact sheets, `all-frames/second-000.jpg` through `second-087.jpg`. Each sheet contains 30 consecutive frames; the last contains 23 frames. All 13 scene transitions are included. Full-size encoded stills were also reviewed in `full-size/`:

| File | Time | Focus |
| --- | --- | --- |
| opening.png | 6.30 s | Larger labels, cell outlines, settled camera |
| context.png | 22.80 s | Civic row and MPG/cylinders/horsepower leaders |
| weight.png | 31.50 s | 4354 close-up, diagnostic cards, tipped scale |
| missing.png | 37.00 s | 0.0 and expects-70 explanation, no crossing connector |
| categories.png | 49.80 s | USA/Japan/Europe outlines and separate leaders |
| japan.png | 54.00 s | Suggestion and categorical technical explanation |
| ledger.png | 75.00 s | Single applied change and ledger explanation |
| history.png | 82.00 s | Undo/reapply history and actual cell value |
| title.png | 86.00 s | Clean closing background and title layout |

Opening camera motion settles before the short dwell and exits without the reported snap. Loading fills left to right before findings. The 4354 camera remains close through the diagnosis; the scale moves and tilts toward the heavier right side. Annotation entrances follow their referenced glyphs. The categorical suggestion and technical card disappear together before the review action, including at 60 seconds. Ledger state matches the visible value and uses one current change card. The final workspace fades completely before title text appears. No additional alignment, unintended overlap, or transition defect was found in this visual review.

## Automated checks

- TypeScript check passed.
- All 2,633 video frames and the audio track decoded successfully.
- All 2,633 frames have layout audit records; zero issues in the defined checks.
- All 17 explanatory reading holds are 54 frames (1.8 seconds).
- All 13 skipped source-time intervals preserve the visible authored motion state; see `timing-continuity.json`.
- Existing scale, loading, categorical entrance/exit, close-up and closing-background assertions passed.

Contact-sheet inspection is a visual review, not an assurance that every possible defect is absent. Layout and continuity assertions cover defined relationships and motion properties; they do not replace visual inspection or soundtrack listening. No new listening review was performed in this revision. Numerical evidence and illustrative categorical provenance are unchanged.
