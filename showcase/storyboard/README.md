# TabLint motion study

This is the authored 72-second, 14-shot storyboard retained as editable input to the final animation. The approved film is [87.77 seconds with sound](../../demo/showcase/README.md); its revised timing and camera transitions are in the [Remotion project](../README.md).

The study uses a light desktop, cell-anchored diagnostics, typed text and rotating button highlights. There is no mouse pointer. Numeric values come from the saved Auto MPG report; the specific categorical suggestion is illustrative. [Provenance](provenance.json).

## View the source

From the repository root:

```sh
python3 -m http.server 7080 --bind 127.0.0.1 --directory showcase/storyboard
```

Open `http://127.0.0.1:7080/index.html`. Space toggles playback; the timeline scrubs the study. Choose a shot and Entry, Action, Exit or Press to inspect it. The viewer computes frames from layered SVGs, so exported duplicate entry/exit files are unnecessary.

## Edit and regenerate

- `shots.json`: source timings, typing, interactions and semantic anchors.
- `assets/*-layers.svg`: editable source art consumed by the video generator.
- `assets/*-action.svg`: shot-picker thumbnails.
- `viewer.js` and `style.css`: interactive study viewer.
- `demo-data/`: attributed categorical example table and answer key.

To regenerate the final animation after edits, follow [the showcase instructions](../README.md). The bundled Inter font retains its [license](assets/FONT-LICENSE.txt).
