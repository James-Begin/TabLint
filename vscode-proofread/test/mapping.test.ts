import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as path from "path";
import { parseCsv, mapIssues } from "../src/mapping";

const fx = (f: string) => path.join(__dirname, "..", "..", "test", "fixtures", f);

test("parseCsv: quotes, escaped quotes, blank lines", () => {
  const { records, multiline } = parseCsv('a,b,c\n\n"x, y",2,"say ""hi"""\n1,,3\n');
  assert.equal(multiline, false);
  assert.equal(records.length, 3);
  assert.deepEqual(records[1].values, ["x, y", "2", 'say "hi"']);
  assert.equal(records[1].line, 2);                        // blank line skipped but physical line kept
  assert.deepEqual(records[2].values, ["1", "", "3"]);
  const t = '"x, y",2';
  assert.equal(t.slice(records[1].fields[1].start, records[1].fields[1].end), "2");
});

test("real report: every issue lands on the recorded value in the CSV", () => {
  const text = fs.readFileSync(fx("breast_dirty.csv"), "utf8");
  const report = JSON.parse(fs.readFileSync(fx("breast_dirty.proofread.json"), "utf8"));
  const { mapped, skipped } = mapIssues(text, report.issues);
  assert.equal(skipped, 0);
  assert.equal(mapped.length, report.issues.length);
  assert.ok(mapped.every(m => m.matchesReport), "every squiggle is on the value the report saw");
  const lines = text.split(/\r?\n/);
  const top = mapped[0];
  assert.equal(lines[top.line].slice(top.start, top.end), top.recordedInFile);
  assert.match(top.message, /TabPFN-3\.5 expects/);
});

test("edited cell no longer matches the report (no stale squiggle)", () => {
  const text = fs.readFileSync(fx("breast_dirty.csv"), "utf8");
  const report = JSON.parse(fs.readFileSync(fx("breast_dirty.proofread.json"), "utf8"));
  const first = mapIssues(text, report.issues).mapped[0];
  const lines = text.split("\n");
  lines[first.line] = lines[first.line].slice(0, first.start) + first.suggestion + lines[first.line].slice(first.end);
  const again = mapIssues(lines.join("\n"), report.issues).mapped[0];
  assert.equal(again.matchesReport, false);
});

test("categorical issue maps to its CSV field with a category-specific explanation", () => {
  const text = "model,origin,weight\nhonda civic,Japna,1800\n";
  const { mapped, skipped } = mapIssues(text, [{
    kind: "category", row: 0, column: "origin", value: "Japna", suggested: "Japan",
    surprise: 2.8, cause: "possible typo of 'Japan' ('Japna' appears 1×)",
  }]);
  assert.equal(skipped, 0);
  assert.equal(mapped.length, 1);
  assert.equal(mapped[0].kind, "category");
  assert.equal(mapped[0].recordedInFile, "Japna");
  assert.equal(text.split("\n")[mapped[0].line].slice(mapped[0].start, mapped[0].end), "Japna");
  assert.equal(mapped[0].suggestion, "Japan");
  assert.equal(mapped[0].matchesReport, true);
  assert.match(mapped[0].message, /unlikely in this row.*suggests "Japan"/);
});
