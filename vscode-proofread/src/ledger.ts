import * as fs from "fs";
import * as path from "path";
import { randomUUID } from "crypto";

export type Decision = "replace" | "dismiss" | "flag" | "undo";
export interface LedgerEvent {
  id: string;
  at: string;
  action: Decision;
  row: number;
  column: string;
  before: string;
  after: string;
  why: string;
  note?: string;
  targetId?: string;
}
export interface Ledger { schema: "proofread-ledger/1"; csv: string; events: LedgerEvent[] }

export const ledgerPath = (csv: string) => csv.replace(/\.csv$/i, "") + ".proofread.ledger.json";

export function readLedger(csv: string): Ledger {
  const file = ledgerPath(csv);
  if (!fs.existsSync(file)) return { schema: "proofread-ledger/1", csv: path.basename(csv), events: [] };
  const data: unknown = JSON.parse(fs.readFileSync(file, "utf8"));
  if (!data || typeof data !== "object" || (data as Ledger).schema !== "proofread-ledger/1" || !Array.isArray((data as Ledger).events)) {
    throw new Error(`Invalid TabLint ledger: ${file}`);
  }
  return data as Ledger;
}

export function newEvent(action: Decision, row: number, column: string, before: string, after: string, why: string, note?: string, targetId?: string): LedgerEvent {
  return { id: randomUUID(), at: new Date().toISOString(), action, row, column, before, after, why, ...(note ? { note } : {}), ...(targetId ? { targetId } : {}) };
}

export function appendEvent(csv: string, event: LedgerEvent): Ledger {
  const ledger = readLedger(csv);
  ledger.events.push(event);
  const file = ledgerPath(csv);
  const temp = `${file}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(temp, JSON.stringify(ledger, null, 2) + "\n", "utf8");
    fs.renameSync(temp, file);
  } catch (error) {
    if (fs.existsSync(temp)) fs.unlinkSync(temp);
    throw error;
  }
  return ledger;
}

export function undoneIds(ledger: Ledger): Set<string> {
  return new Set(ledger.events.filter(e => e.action === "undo" && e.targetId).map(e => e.targetId!));
}

export function isDismissed(ledger: Ledger, row: number, column: string, value?: string): boolean {
  const undone = undoneIds(ledger);
  return ledger.events.some(e => e.action === "dismiss" && e.row === row && e.column === column && (value === undefined || e.before === value) && !undone.has(e.id));
}
