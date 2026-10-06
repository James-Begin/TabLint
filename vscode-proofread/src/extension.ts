import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { exec } from "child_process";
import { mapIssues, Mapped, Issue, parseCsv } from "./mapping";
import { appendEvent, isDismissed, Ledger, LedgerEvent, newEvent, readLedger } from "./ledger";

const collection = vscode.languages.createDiagnosticCollection("proofread");
const fixes = new Map<string, Map<string, Mapped>>();   // uri -> "line:start" -> mapped issue
let status: vscode.StatusBarItem;
let ledgerPanel: vscode.WebviewPanel | undefined;
let ledgerCsv: string | undefined;

const isCsv = (doc: vscode.TextDocument) => doc.uri.scheme === "file" && doc.fileName.toLowerCase().endsWith(".csv");
const sidecar = (file: string) => file.replace(/\.csv$/i, "") + ".proofread.json";

function load(doc: vscode.TextDocument, announce = false): void {
  if (!isCsv(doc)) return;
  const p = sidecar(doc.fileName);
  if (!fs.existsSync(p)) { if (announce) vscode.window.showInformationMessage("TabLint: no saved report next to this CSV. Run “TabLint: Check this CSV”."); return; }
  let report: any;
  try { report = JSON.parse(fs.readFileSync(p, "utf8")); } catch { vscode.window.showWarningMessage(`TabLint: could not read ${path.basename(p)}`); return; }
  if (report.schema !== "proofread/1") return;
  const { mapped, skipped, multiline } = mapIssues(doc.getText(), report.issues ?? []);
  let ledger: Ledger;
  try { ledger = readLedger(doc.fileName); } catch (error) { vscode.window.showWarningMessage(String(error)); return; }
  const diags: vscode.Diagnostic[] = []; const byKey = new Map<string, Mapped>();
  let stale = 0;
  for (const m of mapped) {
    if (!m.matchesReport) { stale++; continue; }          // cell edited since the report: don't squiggle it
    const issue = report.issues[m.issueIndex] as Issue;
    if (isDismissed(ledger, issue.row, issue.column, m.recordedInFile)) continue;
    const range = new vscode.Range(m.line, m.start, m.line, m.end);
    const d = new vscode.Diagnostic(range, m.message, m.kind === "label" ? vscode.DiagnosticSeverity.Information : vscode.DiagnosticSeverity.Warning);
    d.source = "TabLint (TabPFN-3.5)"; d.code = m.kind;
    diags.push(d); byKey.set(`${m.line}:${m.start}`, m);
  }
  collection.set(doc.uri, diags); fixes.set(doc.uri.toString(), byKey);
  status.text = `$(eye) TabLint: ${diags.length} issue${diags.length === 1 ? "" : "s"}`; status.show();
  const notes = [skipped && `${skipped} issues could not be placed`, stale && `${stale} cells changed since the report`,
                 multiline && "quoted multi-line fields are not supported"].filter(Boolean);
  if (announce || notes.length) vscode.window.showInformationMessage(`TabLint: ${diags.length} issues${notes.length ? " (" + notes.join("; ") + ")" : ""}.`);
}

function issueFor(doc: vscode.TextDocument, m: Mapped): Issue | undefined {
  try { return JSON.parse(fs.readFileSync(sidecar(doc.fileName), "utf8")).issues?.[m.issueIndex] as Issue; }
  catch { return undefined; }
}

function explanation(issue: Issue): string {
  return `${issue.cause}${issue.evidence ? `; ${issue.evidence}` : ""}. TabPFN-3.5 surprise ${issue.surprise.toFixed(1)}.`;
}

function refreshLedger(): void {
  if (ledgerPanel && ledgerCsv) {
    try { ledgerPanel.webview.html = ledgerHtml(readLedger(ledgerCsv)); }
    catch (error) { vscode.window.showErrorMessage(String(error)); }
  }
}

async function applyFix(uri: vscode.Uri, m: Mapped): Promise<void> {
  const doc = await vscode.workspace.openTextDocument(uri);
  const issue = issueFor(doc, m); if (!issue) { vscode.window.showWarningMessage("TabLint: reload the report before applying this fix."); return; }
  const range = new vscode.Range(m.line, m.start, m.line, m.end);
  const before = doc.getText(range);
  const quoted = `"${m.recordedInFile.replace(/"/g, '""')}"`;
  if (before !== m.recordedInFile && before !== quoted) { vscode.window.showWarningMessage("TabLint: this cell changed; reload the report before fixing it."); return; }
  const after = before === quoted ? `"${m.suggestion.replace(/"/g, '""')}"` : m.suggestion;
  const edit = new vscode.WorkspaceEdit(); edit.replace(uri, range, after);
  if (!await vscode.workspace.applyEdit(edit)) return;
  try {
    appendEvent(doc.fileName, newEvent("replace", issue.row, issue.column, before, after, explanation(issue)));
  } catch (error) {
    const rollback = new vscode.WorkspaceEdit(); rollback.replace(uri, new vscode.Range(m.line, m.start, m.line, m.start + after.length), before);
    await vscode.workspace.applyEdit(rollback);
    vscode.window.showErrorMessage(`TabLint: fix was reverted because the ledger could not be saved: ${error}`);
    return;
  }
  load(doc); refreshLedger();
}

async function decide(uri: vscode.Uri, m: Mapped, action: "dismiss" | "flag"): Promise<void> {
  const doc = await vscode.workspace.openTextDocument(uri);
  const issue = issueFor(doc, m); if (!issue) return;
  let note: string | undefined;
  if (action === "flag") {
    note = await vscode.window.showInputBox({ prompt: "Why flag this value for review?", placeHolder: "Optional note for the ledger" });
    if (note === undefined) return;
  }
  try { appendEvent(doc.fileName, newEvent(action, issue.row, issue.column, m.recordedInFile, m.recordedInFile, explanation(issue), note)); }
  catch (error) { vscode.window.showErrorMessage(`TabLint: could not save decision: ${error}`); return; }
  load(doc); refreshLedger();
}

async function undoDecision(csv: string, id: string): Promise<void> {
  let ledger: Ledger;
  try { ledger = readLedger(csv); } catch (error) { vscode.window.showErrorMessage(String(error)); return; }
  const event = ledger.events.find(e => e.id === id && e.action !== "undo");
  if (!event || ledger.events.some(e => e.action === "undo" && e.targetId === id)) return;
  const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(csv));
  if (event.action === "replace") {
    const { records } = parseCsv(doc.getText());
    const col = records[0]?.values.indexOf(event.column) ?? -1;
    const rec = records[event.row + 1];
    if (!rec || col < 0 || !rec.fields[col] || doc.getText(new vscode.Range(rec.line, rec.fields[col].start, rec.line, rec.fields[col].end)) !== event.after) {
      vscode.window.showWarningMessage("TabLint: the cell has changed since this fix; undo would overwrite a newer value."); return;
    }
    const field = rec.fields[col];
    const edit = new vscode.WorkspaceEdit(); edit.replace(doc.uri, new vscode.Range(rec.line, field.start, rec.line, field.end), event.before);
    if (!await vscode.workspace.applyEdit(edit)) return;
    try { appendEvent(csv, newEvent("undo", event.row, event.column, event.after, event.before, "Restored the value before the TabLint fix.", undefined, id)); }
    catch (error) {
      const rollback = new vscode.WorkspaceEdit(); rollback.replace(doc.uri, new vscode.Range(rec.line, field.start, rec.line, field.start + event.before.length), event.after);
      await vscode.workspace.applyEdit(rollback);
      vscode.window.showErrorMessage(`TabLint: undo was reverted because the ledger could not be saved: ${error}`); return;
    }
  } else {
    try { appendEvent(csv, newEvent("undo", event.row, event.column, event.before, event.after, `Reversed ${event.action} decision.`, undefined, id)); }
    catch (error) { vscode.window.showErrorMessage(`TabLint: could not save undo: ${error}`); return; }
  }
  load(doc); refreshLedger();
}

const htmlEscape = (value: string): string => value.replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]!));
function ledgerHtml(ledger: Ledger): string {
  const undone = new Set(ledger.events.filter(e => e.action === "undo").map(e => e.targetId));
  const rows = ledger.events.filter(e => e.action !== "undo").slice().reverse().map((e: LedgerEvent) => {
    const reversed = undone.has(e.id);
    const label = e.action === "replace" ? `${e.before} → ${e.after}` : e.action === "flag" ? "Flagged for review" : "Dismissed flag";
    return `<article><div class="top"><span class="tag ${htmlEscape(e.action)}">${htmlEscape(e.action)}</span><span class="date">${htmlEscape(new Date(e.at).toLocaleString())}</span></div><h2>CSV line ${e.row + 2} · ${htmlEscape(e.column)}</h2><div class="value">${htmlEscape(label)}</div><p><b>Why</b> ${htmlEscape(e.why)}</p>${e.note ? `<p><b>Note</b> ${htmlEscape(e.note)}</p>` : ""}<div class="bottom"><span>${reversed ? "Undone" : e.action === "replace" ? "Applied" : e.action === "flag" ? "Needs review" : "Dismissed"}</span>${reversed ? "" : `<button data-undo="${htmlEscape(e.id)}">Undo</button>`}</div></article>`;
  }).join("");
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"><style>body{font:14px var(--vscode-font-family);color:var(--vscode-foreground);padding:20px;background:var(--vscode-editor-background)}h1{font-size:24px;margin:0 0 5px}header p{color:var(--vscode-descriptionForeground);margin:0 0 22px}article{border:1px solid var(--vscode-panel-border);border-radius:9px;padding:17px;margin:0 0 13px;background:var(--vscode-sideBar-background)}.top,.bottom{display:flex;align-items:center;justify-content:space-between;gap:12px}.tag{text-transform:uppercase;font-size:11px;font-weight:700;letter-spacing:.09em;color:var(--vscode-charts-blue)}.tag.flag{color:var(--vscode-charts-orange)}.tag.dismiss{color:var(--vscode-descriptionForeground)}.date{font-size:11px;color:var(--vscode-descriptionForeground)}h2{font-size:16px;margin:12px 0 6px}.value{font:600 20px var(--vscode-editor-font-family)}p{line-height:1.5;color:var(--vscode-descriptionForeground)}p b{color:var(--vscode-foreground)}.bottom{border-top:1px solid var(--vscode-panel-border);padding-top:12px;font-size:12px;font-weight:600}button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;border-radius:4px;padding:7px 13px;cursor:pointer}button:hover{background:var(--vscode-button-hoverBackground)}.empty{padding:25px;border:1px dashed var(--vscode-panel-border);color:var(--vscode-descriptionForeground)}</style><header><h1>TabLint ledger</h1><p>${htmlEscape(ledger.csv)} · Each decision stays recorded, even after undo.</p></header>${rows || '<div class="empty">No TabLint decisions yet. Apply, flag, or dismiss a suggestion to start the ledger.</div>'}<script>const vscode=acquireVsCodeApi();document.querySelectorAll('[data-undo]').forEach(b=>b.addEventListener('click',()=>vscode.postMessage({type:'undo',id:b.dataset.undo})));</script></html>`;
}

function showLedger(): void {
  const ed = vscode.window.activeTextEditor;
  if (!ed || !isCsv(ed.document)) { vscode.window.showWarningMessage("TabLint: open a .csv file first."); return; }
  ledgerCsv = ed.document.fileName;
  if (!ledgerPanel) {
    ledgerPanel = vscode.window.createWebviewPanel("proofreadLedger", "TabLint ledger", vscode.ViewColumn.Beside, { enableScripts: true });
    ledgerPanel.onDidDispose(() => { ledgerPanel = undefined; ledgerCsv = undefined; });
    ledgerPanel.webview.onDidReceiveMessage(message => { if (message?.type === "undo" && typeof message.id === "string" && ledgerCsv) void undoDecision(ledgerCsv, message.id); });
  } else ledgerPanel.reveal(vscode.ViewColumn.Beside);
  refreshLedger();
}

async function check(): Promise<void> {
  const ed = vscode.window.activeTextEditor;
  if (!ed || !isCsv(ed.document)) { vscode.window.showWarningMessage("TabLint: open a .csv file first."); return; }
  await ed.document.save();
  const cfg = vscode.workspace.getConfiguration("proofread");
  let label = cfg.get<string>("labelColumn", "");
  if (!label) {
    const header = ed.document.lineAt(0).text.split(",").map(s => s.replace(/^"|"$/g, ""));
    const pick = await vscode.window.showQuickPick(["(no label column)", ...header], { placeHolder: "Label column to check (optional)" });
    if (pick === undefined) return;
    label = pick === "(no label column)" ? "" : pick;
  }
  const file = ed.document.fileName; const report = sidecar(file);
  const q = (s: string) => `"${s.replace(/"/g, '\\"')}"`;
  const cmd = `${cfg.get<string>("command", "uv run proofread")} check ${q(file)} --report ${q(report)} --out ${q(file.replace(/\.csv$/i, "") + ".proofread")}` +
              ` --threshold ${cfg.get<number>("threshold", 2)}${label ? " --label " + q(label) : ""}` +
              `${cfg.get<boolean>("categorical", true) ? "" : " --no-categorical"}`;
  const cwd = vscode.workspace.getWorkspaceFolder(ed.document.uri)?.uri.fsPath ?? path.dirname(file);
  await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "TabLint: TabPFN-3.5 is reading the table…", cancellable: false },
    () => new Promise<void>((resolve) => {
      exec(cmd, { cwd, maxBuffer: 64 * 1024 * 1024 }, (err, _out, stderr) => {
        if (err) vscode.window.showErrorMessage(`TabLint failed: ${(stderr || err.message).split("\n").filter(Boolean).slice(-3).join(" ")}`);
        else load(ed.document, true);
        resolve();
      });
    }));
}

class Fixer implements vscode.CodeActionProvider {
  provideCodeActions(doc: vscode.TextDocument, _r: vscode.Range, ctx: vscode.CodeActionContext): vscode.CodeAction[] {
    const out: vscode.CodeAction[] = []; const byKey = fixes.get(doc.uri.toString());
    for (const d of ctx.diagnostics.filter(d => d.source === "TabLint (TabPFN-3.5)")) {
      const m = byKey?.get(`${d.range.start.line}:${d.range.start.character}`); if (!m) continue;
      const fix = new vscode.CodeAction(`TabLint: replace with ${m.suggestion}`, vscode.CodeActionKind.QuickFix);
      fix.command = { command: "proofread.applyFix", title: "replace", arguments: [doc.uri, m] }; fix.diagnostics = [d]; fix.isPreferred = true;
      const dismiss = new vscode.CodeAction("TabLint: dismiss this flag", vscode.CodeActionKind.QuickFix);
      dismiss.command = { command: "proofread.dismiss", title: "dismiss", arguments: [doc.uri, m] }; dismiss.diagnostics = [d];
      const flag = new vscode.CodeAction("TabLint: flag for review", vscode.CodeActionKind.QuickFix);
      flag.command = { command: "proofread.flag", title: "flag", arguments: [doc.uri, m] }; flag.diagnostics = [d];
      out.push(fix, flag, dismiss);
    }
    return out;
  }
}

export function activate(context: vscode.ExtensionContext): void {
  status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  status.command = "proofread.check";
  context.subscriptions.push(collection, status,
    vscode.commands.registerCommand("proofread.check", check),
    vscode.commands.registerCommand("proofread.reload", () => vscode.window.activeTextEditor && load(vscode.window.activeTextEditor.document, true)),
    vscode.commands.registerCommand("proofread.clear", () => { collection.clear(); status.hide(); }),
    vscode.commands.registerCommand("proofread.ledger", showLedger),
    vscode.commands.registerCommand("proofread.applyFix", applyFix),
    vscode.commands.registerCommand("proofread.undoDecision", undoDecision),
    vscode.commands.registerCommand("proofread.dismiss", (uri: vscode.Uri, m: Mapped) => decide(uri, m, "dismiss")),
    vscode.commands.registerCommand("proofread.flag", (uri: vscode.Uri, m: Mapped) => decide(uri, m, "flag")),
    vscode.languages.registerCodeActionsProvider({ scheme: "file", pattern: "**/*.csv" }, new Fixer(), { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }),
    vscode.workspace.onDidOpenTextDocument(doc => load(doc)),
    vscode.workspace.onDidSaveTextDocument(doc => load(doc)),
    vscode.window.onDidChangeActiveTextEditor(ed => { if (ed && isCsv(ed.document)) load(ed.document); else status.hide(); }));
  vscode.workspace.textDocuments.forEach(doc => load(doc));
}

export function deactivate(): void { collection.dispose(); }
