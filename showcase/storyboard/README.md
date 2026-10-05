# TabLint — In the flow

Revision 07: 72-second light-mode motion storyboard, 14 shots, 42 keyframes.

Frame 02 labels sit immediately above/below the Impala fields, with short stems. Frame 05 has no extra analysis badge or decorative scan rail. Frame 10 places each category label beside its own row, joined by a short horizontal line.

Numeric and category connectors resolve to exact CSV fields. Focus boxes and diagnostic squiggles use the target glyph bounds after camera transforms. Scene 09 corrects 0.0 to 70 directly in the CSV; detached correction and undo cards are removed. The category view includes actual USA, Japan, and Europe rows.

Text streams left to right in fixed positions. Selected controls have a clockwise perimeter highlight. Most transitions last 350–450ms; the control highlight makes one pass in 650ms. No mouse or cursor appears.

Serve the viewer from the workspace:

```sh
python3 -m http.server 7080 --bind 127.0.0.1 --directory outputs/proofread-storyboard-v2
```

Open http://127.0.0.1:7080/index.html. Space toggles playback; the timeline scrubs globally. Choose a shot and Entry / Action / Exit / Press; Loop shot repeats its timing. Every film text node has a clip mask and editable typing timing in shots.json.

See STORYBOARD.md for all frames, PRODUCTION.md for the animation specification, RESEARCH.md for reference evidence, and VERIFICATION.md for checks. The contact sheet and per-shot SVGs/PNGs provide shareable visuals. Inter is bundled under the included font license.

The specific categorical suggestion remains a planning example pending measured inference. Numeric examples use the existing saved report. This is the storyboard and motion study, not the final film render.

Regenerate SVGs and JSON with `python3 work/build_storyboard_v2.py`. The viewer and prose remain separately editable.

Scene 13 replaces the bottom green message with a white callout pointing to the ledger: view changes and their reasons, revert a fix, or flag for review.
