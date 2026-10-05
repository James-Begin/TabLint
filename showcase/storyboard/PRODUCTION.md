# TabLint — animation direction

72 seconds · 1920×1080 · 30 fps · 14 shots · 42 keyframes · light desktop

The approved direction is **TabLint**, with no mouse or cursor in any frame. Use selection and press feedback on the real control to explain each action. This directory is the revised motion storyboard and timing study; the finished video remains a separate production step.

## Interaction grammar

All selections keep the control upright. A bright teal segment travels clockwise around its rounded outline for 650ms, with a soft tint and a small press compression. The action completes after the highlight's pass. Focused diagnostic values use a stationary outline. No mouse, cursor, hand, pointer trail, or floating click ring appears.

The reapply menu occupies the left editor pane, and its highlight inherits the same transform. It does not cover the decision ledger. CSV values and ledger states switch cleanly; different values never crossfade on top of each other.

## Typed text and reading holds

Every SVG text node has a frame-driven left-to-right mask. The full line remains laid out at its final size and position, so typing never shifts numbers, centered headings, card geometry, or neighboring text. The browser measures actual glyph widths, then reveals prefixes in discrete character steps. No blinking caret is added.

Foreground lines reveal at approximately 48 characters per second with 120ms stagger; long CSV lines use compact parallel reveals with 45ms row stagger. Repeated workspace text continues from its original appearance instead of resetting every shot. New messages and state labels type when their owner appears. The timeline reserves reading time after a reveal.

`shots.json` stores each text node, mask, absolute start, and duration, plus the control perimeter and orbit timing. The same frame-driven values must be used by the final Remotion components. Browser requestAnimationFrame is only for the interactive storyboard.

## Timing and layout

Most travel now takes **350–450ms**, with 400ms layer transitions and a 350ms scene-entry adjustment. Controls make one 650ms perimeter pass. Keep smooth deceleration and stable reading holds, without slow drift or spring wobble. The 72-second timing includes extra time for typed text and review.

Numeric corrections happen in the actual CSV field. The standalone red 0.0 and detached undo cards are removed; the reason receipt retains its lower lane. Diagnostic panels sit clear of their active fields, the interval diagram has its own lower lane, and the undo note stays clear of the CSV. The CSV is clipped to its own pane when the ledger is open. Do not let new menus cover the history being explained.

## Continuous composition

Keep a shared desktop/editor coordinate system. The weight token matches across Impala and Civic. Numeric context cards point to the real Civic mpg, cylinders, and horsepower fields. Numeric and category file changes are signalled by tab selection. Category and numeric ledgers belong to separate files; maintain the panel geometry during that content switch.

The preview has a 350ms scene-entry adjustment plus independent vector-layer timing. The final Remotion composition must use matched camera endpoints and shared UI components so those handoffs remain continuous. Do not turn these panels back into flattened-image pans.

Use shared `Desktop`, `CsvEditor`, `FocusFrame`, `ControlPress`, `NativePalette`, `DiagnosticHover`, `QuickFix`, `CellValue`, `Ledger`, and `EditorialOverlay` components. Drive them from `useCurrentFrame()` and the event timing in `shots.json`. Browser requestAnimationFrame is only for this preview; exported animation must use frame-driven transforms.

## Product fidelity and data

The VS Code extension's visible titles, diagnostic source, action labels, notifications, status, and ledger are renamed TabLint. Existing command IDs, CLI invocation, configuration keys, saved-report schemas, and sidecar paths remain compatible. The storyboard shows the new visible names.

Numeric evidence comes from the existing saved report. Civic expected weight 1877 is a model expectation, not the verified original 1795; inspect this example without treating a suggestion as ground truth. Rabbit original horsepower 70 is verified by the planted-error answer key, so this is the applied-fix example.

Categorical checking is implemented. The planned origin example checks the categorical CSV with no label, which distinguishes categorical-cell detection from label auditing. The specific Toyota USA→Japan suggestion needs a measured `kind: category` report before final production. Do not invent probabilities or reasons; bind the actual result, or replace the example if it is not flagged.

The shortened analysis wait uses the native progress notification; no extra analysis status badge appears. The ledger preserves original values and reasons; undo keeps the original event, and reapply adds a new Applied event above the Undone event. Simplified storyboard chrome and condensed reasons must become faithful native details in the finished film.

## Deliverables

- `index.html`: playable board with scrubber, loop-shot control, and entry/action/exit selection.
- `shots.json`: timing, independent layers, control interactions, camera directions, transitions, and voiceover.
- `assets/shot-XX-layers.svg`: editable scenes without pointer artwork.
- `assets/shot-XX-entry.svg`, `-action.svg`, `-exit.svg`: every revised keyframe.
- `contact-sheet.svg` and `.png`: six representative scenes.
- `demo-data/auto_mpg_categories.csv` and `provenance.json`: prepared input and evidence requirements.

## Exact field anchoring — revision 06

Each CSV token is tagged with its line number and column name. Runtime annotations use the rendered token bounds and the complete camera/layer transforms on every draw; no guessed coordinates or substring search determine the target. Static SVG keyframes use matching field offsets and font metrics.

- Civic: CSV line 183, mpg 33.0, cylinders 4, horsepower 53.0, weight 4354.0.
- Rabbit: CSV line 177, horsepower 0.0; its preceding displacement 90.0 is a different field. Apply and undo highlight the actual horsepower field as its width changes.
- Categories: Toyota line 16 USA, Datsun line 20 Japan, Volkswagen line 21 Europe. The category viewport now includes all three native rows.

Connectors end at the edge of the corresponding token box, clear of the glyphs. Diagnostic squiggles span only their value. Keep this geometry contract in the final film.

## Label placement — revision 06

Scene 02: car name and horsepower labels sit above their Impala fields; weight sits below. Short stems connect each label to its token outline. Label placement resolves from the same transformed token bounds as the focus boxes, so camera and row travel never detach the annotations.

Scene 05: retain the native progress notification, then reveal the exact diagnostic squiggles. Remove the extra status badge and decorative scan rail.

Scene 10: labels sit in the free space to the right of their own rows, level with their origin values. Each connector is a short horizontal segment ending at its token outline. The three paths have no shared trunk, screen-edge rail, or crossings.

## Ledger explanation — revision 07

Scene 13 uses a white explanation box beneath the CSV rows, with a short leader attached to the left edge of the visible ledger. It explains viewing all changes and reasons, reverting fixes, and flagging values for review. Remove the former green history message. The callout types in early, holds through undo/reapply, and clears before the final scene. Keep the rows, ledger records, and controls unobscured.
