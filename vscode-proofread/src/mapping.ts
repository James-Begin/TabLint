// Pure logic (no VS Code imports): map TabLint issues onto character ranges in CSV text.

export interface FieldRange { start: number; end: number }
export interface CsvRecord { line: number; fields: FieldRange[]; values: string[] }

/** Split CSV text into records with the character range of each field. Blank lines are skipped (as pandas does).
 *  Quoted fields are supported; quoted fields containing line breaks are not (returns `multiline: true`). */
export function parseCsv(text: string, delimiter = ","): { records: CsvRecord[]; multiline: boolean } {
  const lines = text.split(/\r?\n/);
  const records: CsvRecord[] = [];
  let multiline = false;
  lines.forEach((line, ln) => {
    if (line.trim() === "") return;
    const fields: FieldRange[] = []; const values: string[] = [];
    let i = 0;
    while (i <= line.length) {
      const start = i;
      let value = "";
      if (line[i] === '"') {
        i++;
        while (i < line.length) {
          if (line[i] === '"' && line[i + 1] === '"') { value += '"'; i += 2; continue; }
          if (line[i] === '"') { i++; break; }
          value += line[i++];
        }
        if (i >= line.length && line[line.length - 1] !== '"') multiline = true;
        while (i < line.length && line[i] !== delimiter) i++;
      } else {
        while (i < line.length && line[i] !== delimiter) value += line[i++];
      }
      fields.push({ start, end: i }); values.push(value);
      i++;
    }
    records.push({ line: ln, fields, values });
  });
  return { records, multiline };
}

export interface Issue {
  kind: "cell" | "label" | "category"; row: number; column: string; value: unknown; suggested: unknown;
  low?: number | null; high?: number | null; surprise: number; cause: string; evidence?: string;
}
export interface Mapped {
  issueIndex: number; line: number; start: number; end: number; message: string; suggestion: string;
  kind: "cell" | "label" | "category"; surprise: number; recordedInFile: string; matchesReport: boolean;
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function fmt(v: unknown): string {
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    return Number(v.toPrecision(6)).toString();
  }
  return String(v);
}

/** Map report issues to field ranges. Row r (0-based data row, as in pandas) is CSV record r + 1 (record 0 = header). */
export function mapIssues(text: string, issues: Issue[]): { mapped: Mapped[]; skipped: number; multiline: boolean } {
  const { records, multiline } = parseCsv(text);
  const header = records[0]?.values ?? [];
  const mapped: Mapped[] = []; let skipped = 0;
  issues.forEach((iss, k) => {
    const rec = records[iss.row + 1]; const col = header.indexOf(String(iss.column));
    if (!rec || col < 0 || col >= rec.fields.length) { skipped++; return; }
    const f = rec.fields[col]; const recorded = rec.values[col];
    const numeric = typeof iss.value === "number" && recorded.trim() !== "" && Number.isFinite(Number(recorded));
    const matchesReport = numeric ? Math.abs(Number(recorded) - (iss.value as number)) <= 1e-6 * Math.max(1, Math.abs(iss.value as number))
                                  : recorded === String(iss.value);
    const suggestion = iss.kind === "cell" ? fmt(Number((iss.suggested as number).toPrecision(4))) : String(iss.suggested);
    const range = iss.kind === "cell" && iss.low != null && iss.high != null ? ` (80%: ${fmt(Number(iss.low.toPrecision(4)))}–${fmt(Number(iss.high.toPrecision(4)))})` : "";
    const message = iss.kind === "cell"
      ? `TabLint: ${recorded} looks wrong for this row. TabPFN-3.5 expects ≈ ${suggestion}${range}. ${cap(iss.cause)}.` +
        (iss.evidence ? ` Related: ${iss.evidence}.` : "") + ` Surprise ${iss.surprise.toFixed(1)}.`
      : iss.kind === "category"
      ? `TabLint: "${recorded}" is unlikely in this row; TabPFN-3.5 suggests "${suggestion}". ${cap(iss.cause)}.`
      : `TabLint: label "${recorded}" is unlikely here; TabPFN-3.5 predicts "${suggestion}" (${iss.cause}).`;
    mapped.push({ issueIndex: k, line: rec.line, start: f.start, end: f.end, message, suggestion, kind: iss.kind,
                  surprise: iss.surprise, recordedInFile: recorded, matchesReport });
  });
  return { mapped, skipped, multiline };
}
