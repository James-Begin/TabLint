# TabLint for VS Code: spellcheck for tables

Open a CSV and TabLint underlines cells that are implausible *for their row*, according to TabPFN-3.5, just like
a spellchecker.

- **Hover** a squiggle to see what TabPFN-3.5 expects and why the value was flagged. Numeric cells include an 80%
  plausible range and may name a likely slip (×10, ÷10, swapped digits, 0-for-missing…). Categorical cells show the
  suggested level and may include a typo hint.
- **Quick fix** (💡 or `Ctrl/Cmd + .`): replace the value, flag it for source review, or dismiss the warning.
- The **Problems panel** lists numeric cells, categorical cells, and labels flagged by the saved report.
- **TabLint: Open decision ledger** opens a side panel with each TabLint decision, its model reason, and an Undo button.

## How it works
- The extension reads a saved report next to the CSV (`data.csv` → `data.proofread.json`), so viewing needs no model
  and no GPU.
- **TabLint: Check this CSV with TabPFN-3.5** runs the TabLint CLI (`uv run proofread check …` by default; see
  `proofread.command`) in your workspace and loads the result.
- Categorical-cell checks are enabled by default. The CLI's `--no-categorical` option can skip them when generating a report outside the extension.
- Cells you've edited since the report are not squiggled.
- Fixes, flags, dismissals, and reversals are appended to `data.proofread.ledger.json` beside the CSV. The ledger records the old and new value, row, column, timestamp, model explanation, and any note entered when flagging. Undo restores a fix only if the cell still has the value TabLint wrote, so it cannot overwrite a newer manual edit. Dismissals persist across reloads for that reported value.

## Settings
- `proofread.command`: how to run the CLI (default `uv run proofread`).
- `proofread.threshold`: flag cells with surprise ≥ this value. 2 gives about 87% precision and 57% recall; 3 gives
  about 94% and 20% on the retained benchmark scope (see `docs/PROOFREAD_RESULTS.md` in the repository).
- `proofread.labelColumn`: optional label column to check.
- `proofread.categorical`: include categorical-cell checks (default `true`). Set `false` to reproduce the numeric-only saved Auto MPG demo report.

## Limitations
- Plain comma-separated files with one record per line. Quoted fields may contain commas, but not line breaks.
- A flag is a prompt to check the source, not proof of an error.
