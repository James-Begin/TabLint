import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { appendEvent, isDismissed, ledgerPath, newEvent, readLedger, undoneIds } from "../src/ledger";

test("ledger persists decisions and undo history beside the CSV", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "proofread-ledger-"));
  try {
    const csv = path.join(dir, "cars.csv");
    fs.writeFileSync(csv, "car,horsepower\nRabbit,0\n");
    const fix = newEvent("replace", 0, "horsepower", "0", "70", "Possible missing value recorded as 0; displacement = 90.");
    const flag = newEvent("flag", 0, "horsepower", "70", "70", "Check against source.", "Verify in manual");
    appendEvent(csv, fix);
    appendEvent(csv, flag);
    assert.equal(readLedger(csv).events.length, 2);
    assert.equal(readLedger(csv).events[0].why, fix.why);
    appendEvent(csv, newEvent("undo", 0, "horsepower", "70", "0", "Restored earlier value.", undefined, fix.id));
    const ledger = readLedger(csv);
    assert.equal(ledger.events.length, 3);
    assert.ok(undoneIds(ledger).has(fix.id));
    assert.ok(!undoneIds(ledger).has(flag.id));
    assert.equal(JSON.parse(fs.readFileSync(ledgerPath(csv), "utf8")).schema, "proofread-ledger/1");
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("dismissal is reversed by an undo event", () => {
  const csv = "/tmp/cars.csv";
  const dismiss = newEvent("dismiss", 4, "weight", "4354", "4354", "Review later");
  const ledger = { schema: "proofread-ledger/1" as const, csv, events: [dismiss] };
  assert.ok(isDismissed(ledger, 4, "weight"));
  ledger.events.push(newEvent("undo", 4, "weight", "4354", "4354", "Reopened", undefined, dismiss.id));
  assert.ok(!isDismissed(ledger, 4, "weight"));
});
