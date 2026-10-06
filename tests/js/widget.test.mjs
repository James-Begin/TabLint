// Renders the real proofread/static/widget.js in jsdom with a mock anywidget model; simulates clicks and keys.
import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><div id=el></div>");
globalThis.document = dom.window.document;
const { default: widget } = await import("../../proofread/static/widget.js");

function mockModel(state) {
  const handlers = {}; let saves = 0;
  return {
    get: (k) => state[k],
    set: (k, v) => { state[k] = v; (handlers[`change:${k}`] || []).forEach((f) => f()); },
    on: (ev, f) => (handlers[ev] ||= []).push(f), off: () => {}, save_changes: () => { saves++; },
    get saves() { return saves; },
  };
}
const state = () => ({
  columns: ["a", "b", "y"], only_flagged: false, selected: 0, meta: { n: 4, model: "TabPFN-3.5", patterns: [] },
  rows: [0, 1, 2, 3].map((i) => ({ i, cells: [[1, 2, 30, 4][i], [10, 0, 30, 40][i], ["x", "y", "x", "x"][i]] })),
  issues: [
    { kind: "cell", row: 2, column: "a", value: 30, suggested: 3.0412, low: 2.5, high: 3.5, surprise: 4.2, cause: "possible decimal slip (×10)", evidence: "b = 30" },
    { kind: "cell", row: 1, column: "b", value: 0, suggested: 20, low: 15, high: 25, surprise: 3.1, cause: "possible missing value recorded as 0", evidence: "" },
    { kind: "label", row: 3, column: "y", value: "x", suggested: "y", low: null, high: null, surprise: 2.0, cause: "model gives 1%", evidence: "" },
  ],
  status: ["open", "open", "open"],
});

test("renders flagged cells and the detail panel", () => {
  const el = document.getElementById("el"); el.innerHTML = "";
  widget.render({ model: mockModel(state()), el });
  assert.equal(el.querySelectorAll("td.pr-flag").length, 2);
  assert.equal(el.querySelectorAll("td.pr-label").length, 1);
  const side = el.querySelector(".pr-side").textContent;
  assert.match(side, /Issue 1\/3/); assert.match(side, /3\.041/); assert.match(side, /80% plausible: 2\.5 – 3\.5/);
});

test("accept, dismiss, undo via buttons and keys update the model", () => {
  const el = document.getElementById("el"); el.innerHTML = "";
  const s = state(); const m = mockModel(s);
  widget.render({ model: m, el });
  el.querySelector(".pr-btn-accept").click();
  assert.deepEqual(s.status, ["accepted", "open", "open"]); assert.equal(s.selected, 1);
  assert.equal(el.querySelector("td.pr-accepted").textContent, "3.041");      // shows the accepted value
  el.querySelector(".pr-root").dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "d" }));
  assert.deepEqual(s.status, ["accepted", "dismissed", "open"]);
  el.querySelector(".pr-root").dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "u" }));
  assert.deepEqual(s.status, ["accepted", "open", "open"]);
  el.querySelector("td.pr-label").click();
  assert.equal(s.selected, 2);
  assert.ok(m.saves >= 4);
});

test("only-flagged toggle is sent to Python", () => {
  const el = document.getElementById("el"); el.innerHTML = "";
  const s = state(); widget.render({ model: mockModel(s), el });
  const cb = el.querySelector(".pr-toggle input"); cb.checked = true; cb.dispatchEvent(new dom.window.Event("change"));
  assert.equal(s.only_flagged, true);
});
