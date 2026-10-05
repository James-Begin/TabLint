# Demo video: script and shot list (about 2.5 minutes)

**Dataset:** UCI Auto MPG (CC BY 4.0), 398 real cars. It's ideal for a demo because anyone can sanity-check a car.

**Generated files:** `benchmarks/make_video_demos.py` writes everything into `demo/video/`:

| File | Contents |
|---|---|
| `auto_mpg.csv` + `auto_mpg.proofread.json` | the real, untouched table and its report |
| `auto_mpg_dirty.csv` + `auto_mpg_dirty.proofread.json` | the table with 8 scripted errors, and its report |
| `answer_key.json` | every planted error with its story, true value, rank in Proofread's list, z-score, and whether it was caught |

All numbers below were read from those files. If you regenerate, re-check them against `answer_key.json`.

## Setup before recording
- **Terminal:** about 180×40, large font, dark theme. Run `uv sync --extra notebook` once.
- **Saved reports open instantly.** No GPU or model is needed for any shot except the optional live run.
- **VS Code:** `cd vscode-proofread && npm install && npx @vscode/vsce package && code --install-extension proofread-tables-0.1.0.vsix`,
  then open `demo/video/auto_mpg_dirty.csv`. Its report, `auto_mpg_dirty.proofread.json`, sits next to it and loads automatically.
- **Notebook:** `uv run --with jupyterlab jupyter lab examples/video_demo.ipynb`. **Rehearse this once.** The widget has
  been verified in jsdom, in a real kernel and as rendered HTML in Chrome, but not yet inside a live JupyterLab session.

---

## Shot 1 · Hook: a real error in a famous dataset (0:00–0:25)
**Screen:** `uv run proofread view demo/video/auto_mpg.proofread.json`. The cursor lands on issue 1/23.

**Narration:** "This is Auto MPG: 398 real cars, a classic teaching dataset. Its rows also appear in seaborn's `mpg`
dataset. We didn't touch it. Proofread's number-one flag: a 1970 Buick Estate Wagon with a 455-cubic-inch V8, 225 horsepower
and 14 mpg, recorded at 3,086 pounds. That would make it the lightest V8 in the whole table, only about 100 pounds heavier than
the heaviest Ford Pinto in it (2,984 lb). TabPFN-3.5, reading only
this table, expects about 4,471. The published curb weight is around 4,770."

**On screen:** recorded `3086`, TabPFN `4471` (80%: 4311–4680); related columns displacement 455, horsepower 225, mpg 14.

**Accuracy:** say "much closer to the published figure", not "matches". The published 4,762–4,775 lb (UltimateSpecs;
Wikipedia says 4,900–5,000 lb) sits just above TabPFN's 80% range.

## Shot 2 · What it is (0:25–0:40)
**Narration:** "Proofread is a spellcheck for tables. For every cell, TabPFN-3.5 predicts the value from the rest of
the row, with a full probability distribution and no training. Values far outside it get underlined."

**Screen:** press `n` a few times. Show other flags, then say: "Some flags are just unusual, like the Mazda GLC's
46.6 mpg, the highest in the table. A flag is a prompt to check. If it checks out, I dismiss it." Press `d`.

## Shot 3 · Why it beats a z-score (0:40–1:10)
**Screen:** `uv run proofread view demo/video/auto_mpg_dirty.proofread.json`, then jump to row 181, the Honda Civic.

**Narration:** "Now eight typos of the kind people actually make. Here someone pasted a Chevy Impala's weight, 4,354
pounds, into a Honda Civic. For the weight column that's a perfectly normal number: its z-score is 0.7, so a range check
sees nothing. But for a 4-cylinder, 53-horsepower, 33-mpg car it's impossible. TabPFN expects 1,877. The true value is
1,795."

**On screen:** recorded `4354`, TabPFN `1877` (80%: 1720–2034); related: displacement 91, cylinders 4, horsepower 53.

**Two more of these, each invisible to a z-score:**

| Car | Recorded | z-score | TabPFN expects | Truth | Story |
|---|---|---|---|---|---|
| Ford Mustang II | displacement 5.0 | −1.8 | 302.03 | 302 | litres typed instead of cubic inches |
| VW Rabbit | horsepower 0 | −2.69 | 70 | 70 | "possible missing value recorded as 0" |

## Shot 4 · Spellcheck-style fixes (1:10–1:35)
**Screen (terminal):** on the Chevelle (35,040 lb), the cause reads "possible decimal slip (×10): 35040 → 3504 would be
typical". Press `a`. Do the Datsun next (1.7 → "decimal slip (÷10) → 17"), then press `s`.

**Narration:** "It names the likely slip: an extra zero, a dropped decimal, a 0 standing in for missing. Accept, and it
writes a cleaned CSV plus an audit log of every decision."

**Show:** `auto_mpg_dirty.cleaned.csv` and `auto_mpg_dirty.decisions.json`.

## Shot 5 · Where you work: notebook and VS Code (1:35–2:00)
**Notebook** (`examples/video_demo.ipynb`): run the cells, then click a red cell in the widget and accept. Run
`w.cleaned` and show the fixed rows. Mention `df.proofread.view(label="origin")` for your own data. The label check
catches the Toyota Corona Mark II recorded as made in the USA: TabPFN says Japan and gives "USA" 0.7%.

**VS Code:** in `auto_mpg_dirty.csv`, hover the squiggle on `4354.0` to see "TabPFN-3.5 expects ≈ 1877 (80%: 1720–2034)…".
Press Cmd+. and choose "Proofread: replace with 1877". The Problems panel lists all 28 issues.

## Shot 6 · Honest scoreboard and evidence (2:00–2:25)
**Narration:** "Of the eight planted errors, Proofread ranks six in its top seven. The seventh flag is the real Buick.
The wrong-country label is the top label flag. It missed one: a Corolla's 32 mpg typed as 23, which scored 1.95, just
under the default threshold of 2. That's a believable number for that car. That's the trade-off between precision and
recall, and you can set it."

**Screen:** `answer_key.json` (or the notebook table), then `docs/figures/proofread_hero.png`.

**Narration:** "We pre-registered a benchmark: 14 public datasets, 5 seeds, 5 error types. Proofread beats the best of
seven baselines on 13 of 14 (p = 0.0002), and beats Prior Labs' own TabPFN outlier detector on 14 of 14. Along the way we
found three GPU-only bugs in `tabpfn-extensions` and wrote them up with fixes."

## Shot 7 · Close (2:25–2:35)
`uv run proofread check your.csv`, or `df.proofread.view()`. "Proofread: spellcheck for tables, powered by TabPFN-3.5."

---

## Scoreboard (from `answer_key.json`)

| Planted error | Rank | z-score (z-score flags it?) | TabPFN expects | Truth |
|---|---:|---|---|---|
| Toyota Corona model year 1917 (swapped digits) | 1 | −12.45 (yes) | 1972 | 1971 |
| Chevelle weight 35,040 (extra zero) | 2 | 17.6 (yes) | "decimal slip → 3504" | 3504 |
| Datsun 510 acceleration 1.7 (dropped decimal) | 3 | −4.86 (yes) | "decimal slip → 17" | 17.0 |
| VW Rabbit horsepower 0 (missing as 0) | 4 | −2.69 (**no**) | 70 | 70 |
| Mustang II displacement 5.0 (litres) | 5 | −1.8 (**no**) | 302.03 | 302 |
| Honda Civic weight 4,354 (wrong row) | 6 | 0.71 (**no**) | 1877 | 1795 |
| *(real)* Buick Estate Wagon weight 3,086 | 7 | — | 4471 | ≈ 4,770 published |
| Toyota Corona Mark II origin "USA" (label) | 18 (top label flag) | — | Japan | Japan |
| Toyota Corolla mpg 23 (swapped digits) | **missed** (1.95 < 2) | −0.06 (no) | — | 32 |

## Claims to avoid on camera
- Not "detects all errors". Say "7 of 8 planted errors, plus a real one".
- Not "TabPFN knows the real spec". It infers a plausible value from the other rows.
- Not "the Buick figure is wrong in every copy". We only checked UCI, seaborn's `mpg.csv` and Plotly's dataset.
- The planted errors are our own script, so say so: "eight typos we planted".
