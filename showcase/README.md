# TabLint showcase

The approved animation is [TabLint-demo.mp4](../demo/showcase/TabLint-demo.mp4): 87.77 seconds, 1920×1080, 30 fps, H.264 and AAC stereo. This is an animated recreation of the VS Code workflow with original 112 BPM music, not a screen recording. Numerical evidence comes from the saved Auto MPG report; the specific USA → Japan suggestion is illustrative, not a captured inference result. See [provenance](public/tabl-int/provenance.json) and [visual review](review/VISUAL-REVIEW.md).

## Preview and export

Node.js 22+ is recommended. From this directory:

```sh
npm ci
npm run dev
# Select TabLintDemo in Remotion Studio.
npm run check
npm run render
```

Rendering writes the MP4 in `demo/showcase/` and an all-frame layout audit in `review/`. Remotion downloads its rendering browser on first use. The original font licenses are in `public/fonts/`; dependencies retain their own licenses.

## Edit

- `src/TabLintDemo.tsx`: 14 frame-driven scenes and shared camera transitions.
- `src/tabl-int/engine.ts`: typing, semantic anchors, button selection orbits, loading, scale and audits.
- `src/tabl-int/assets.ts` / `timing.ts`: frozen art and source-time mappings.
- `storyboard/`: approved storyboard source.
- `scripts/retime_tablint.py`: persistent revisions and timing regeneration.
- `scripts/make_tablint_score.py`: original score and interaction sounds.

After generator edits, run `npm run regenerate`, then `npm run score`, `npm run check` and `npm run render`. The encoded export review requires Python with Pillow and an `ffmpeg` executable on PATH (or set `FFMPEG` to its path): `python3 scripts/review_export.py`. Review consecutive encoded frames and full-size stills after any changes; passing geometry assertions alone does not establish visual quality.
