// Runs inside a real VS Code instance (via @vscode/test-electron). No mocha; throws on failure.
import * as vscode from "vscode";
import * as path from "path";
import * as assert from "assert";
import * as fs from "fs";
import * as os from "os";
import { readLedger } from "../../src/ledger";

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

export async function run(): Promise<void> {
  const source = path.resolve(__dirname, "..", "..", "..", "test", "fixtures", "breast_dirty.csv");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "proofread-integration-"));
  const csv = path.join(dir, "breast_dirty.csv");
  fs.copyFileSync(source, csv);
  fs.copyFileSync(source.replace(/\.csv$/, ".proofread.json"), csv.replace(/\.csv$/, ".proofread.json"));
  try {
  const doc = await vscode.workspace.openTextDocument(csv);
  await vscode.window.showTextDocument(doc);
  let diags: readonly vscode.Diagnostic[] = [];
  for (let i = 0; i < 50 && diags.length === 0; i++) { await sleep(200); diags = vscode.languages.getDiagnostics(doc.uri); }
  const ours = diags.filter(d => d.source === "TabLint (TabPFN-3.5)");
  assert.strictEqual(ours.length, 80, `expected 80 squiggles, got ${ours.length}`);
  const top = ours.find(d => doc.getText(d.range) === "211.0") ?? ours[0];
  console.log(`[integration] ${ours.length} diagnostics; e.g. line ${top.range.start.line + 1}: "${doc.getText(top.range)}" — ${top.message.slice(0, 110)}`);
  const actions = await vscode.commands.executeCommand<vscode.CodeAction[]>("vscode.executeCodeActionProvider", doc.uri, top.range);
  const fix = actions.find(a => a.title.startsWith("TabLint: replace with"));
  assert.ok(fix && fix.command, "command-backed quick fix offered");
  console.log(`[integration] quick fix: ${fix!.title}`);
  await vscode.commands.executeCommand(fix!.command!.command, ...(fix!.command!.arguments ?? []));
  const replacement = fix!.title.replace("TabLint: replace with ", "");
  const replaced = doc.lineAt(top.range.start.line).text.slice(top.range.start.character, top.range.start.character + replacement.length);
  console.log(`[integration] after fix the cell reads "${replaced}"`);
  assert.ok(fix!.title.endsWith(replaced));
  const ledger = readLedger(csv);
  assert.strictEqual(ledger.events.length, 1);
  assert.strictEqual(ledger.events[0].action, "replace");
  assert.ok(ledger.events[0].why.length > 0);
  await vscode.commands.executeCommand("proofread.undoDecision", csv, ledger.events[0].id);
  assert.strictEqual(doc.getText(top.range), "211.0");
  assert.strictEqual(readLedger(csv).events[1].action, "undo");
  const quote = new vscode.WorkspaceEdit(); quote.replace(doc.uri, top.range, '"211.0"');
  await vscode.workspace.applyEdit(quote);
  await doc.save();
  const quoted = vscode.languages.getDiagnostics(doc.uri).find(d => d.source === "TabLint (TabPFN-3.5)" && d.range.start.line === top.range.start.line && doc.getText(d.range) === '"211.0"');
  assert.ok(quoted, "quoted numeric field remains diagnosable");
  const quotedActions = await vscode.commands.executeCommand<vscode.CodeAction[]>("vscode.executeCodeActionProvider", doc.uri, quoted!.range);
  const quotedFix = quotedActions.find(a => a.title.startsWith("TabLint: replace with"));
  assert.ok(quotedFix?.command, "quoted field has a fix");
  await vscode.commands.executeCommand(quotedFix!.command!.command, ...(quotedFix!.command!.arguments ?? []));
  assert.ok(doc.lineAt(top.range.start.line).text.includes('"21.15"'), "replacement keeps CSV quoting");
  const quotedEvent = readLedger(csv).events.at(-1)!;
  await vscode.commands.executeCommand("proofread.undoDecision", csv, quotedEvent.id);
  assert.strictEqual(doc.getText(quoted!.range), '"211.0"', "undo restores original quoting");

  const categoryCsv = path.join(dir, "category.csv");
  fs.writeFileSync(categoryCsv, "model,origin,weight\nhonda civic,Japna,1800\n");
  fs.writeFileSync(categoryCsv.replace(/\.csv$/, ".proofread.json"), JSON.stringify({
    schema: "proofread/1", issues: [{
      kind: "category", row: 0, column: "origin", value: "Japna", suggested: "Japan",
      surprise: 2.8, cause: "possible typo of 'Japan' ('Japna' appears 1×)", evidence: "",
    }],
  }));
  const categoryDoc = await vscode.workspace.openTextDocument(categoryCsv);
  await vscode.window.showTextDocument(categoryDoc);
  let categoryDiag: vscode.Diagnostic | undefined;
  for (let i = 0; i < 50 && !categoryDiag; i++) {
    await sleep(100);
    categoryDiag = vscode.languages.getDiagnostics(categoryDoc.uri).find(d => d.source === "TabLint (TabPFN-3.5)");
  }
  assert.ok(categoryDiag, "categorical finding appears in VS Code");
  assert.strictEqual(categoryDiag!.code, "category");
  assert.match(categoryDiag!.message, /suggests "Japan"/);
  const categoryActions = await vscode.commands.executeCommand<vscode.CodeAction[]>("vscode.executeCodeActionProvider", categoryDoc.uri, categoryDiag!.range);
  const categoryFix = categoryActions.find(a => a.title === "TabLint: replace with Japan");
  assert.ok(categoryFix?.command, "categorical finding has a quick fix");
  await vscode.commands.executeCommand(categoryFix!.command!.command, ...(categoryFix!.command!.arguments ?? []));
  assert.strictEqual(categoryDoc.getText(categoryDiag!.range), "Japan");
  const categoryEvent = readLedger(categoryCsv).events[0];
  assert.strictEqual(categoryEvent.action, "replace");
  assert.match(categoryEvent.why, /possible typo/);
  await vscode.commands.executeCommand("proofread.undoDecision", categoryCsv, categoryEvent.id);
  assert.strictEqual(categoryDoc.getText(categoryDiag!.range), "Japna");
  assert.strictEqual(readLedger(categoryCsv).events[1].action, "undo");
  console.log("[integration] PASS");
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}
