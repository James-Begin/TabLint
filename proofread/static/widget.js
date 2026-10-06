// Proofread notebook widget (anywidget ESM). Plain DOM, no dependencies.
const fmt = (v) => {
  if (typeof v === "number") return Number.isFinite(v) ? String(Number(v.toPrecision(6))) : String(v);
  return v === null || v === undefined ? "" : String(v);
};
const sig4 = (v) => (typeof v === "number" && Number.isFinite(v) ? String(Number(v.toPrecision(4))) : fmt(v));

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "pr-root"; root.tabIndex = 0;
  el.appendChild(root);
  const history = [];

  const setStatus = (k, s) => {
    const st = [...model.get("status")];
    history.push([k, st[k]]);
    st[k] = s; model.set("status", st);
    const next = st.findIndex((x, i) => i > k && x === "open");
    const wrap = next >= 0 ? next : st.findIndex((x) => x === "open");
    if (wrap >= 0) model.set("selected", wrap);
    model.save_changes();
  };
  const undo = () => {
    const h = history.pop(); if (!h) return;
    const st = [...model.get("status")]; st[h[0]] = h[1];
    model.set("status", st); model.set("selected", h[0]); model.save_changes();
  };
  const move = (d) => {
    const n = model.get("issues").length; if (!n) return;
    model.set("selected", (model.get("selected") + d + n) % n); model.save_changes();
  };

  function draw() {
    const cols = model.get("columns"), rows = model.get("rows"), issues = model.get("issues");
    const status = model.get("status"), sel = model.get("selected"), meta = model.get("meta") || {};
    const byCell = new Map(issues.map((r, k) => [`${r.row}\u0000${r.column}`, k]));
    root.innerHTML = "";

    const head = document.createElement("div"); head.className = "pr-head";
    const counts = ["open", "accepted", "dismissed"].map((s) => `${status.filter((x) => x === s).length} ${s}`).join(" · ");
    head.innerHTML = `<b>TabLint</b> · ${meta.n ?? "?"} rows · ${issues.length} issues · ${counts} <span class="pr-model">${meta.model ?? "TabPFN-3.5"}</span>`;
    const toggle = document.createElement("label"); toggle.className = "pr-toggle";
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = model.get("only_flagged");
    cb.onchange = () => { model.set("only_flagged", cb.checked); model.save_changes(); };
    toggle.append(cb, " only flagged rows"); head.appendChild(toggle);
    root.appendChild(head);

    const body = document.createElement("div"); body.className = "pr-body";
    const wrap = document.createElement("div"); wrap.className = "pr-grid";
    const table = document.createElement("table");
    const thead = table.createTHead().insertRow();
    ["row", ...cols].forEach((c) => { const th = document.createElement("th"); th.textContent = c; thead.appendChild(th); });
    const tb = table.createTBody();
    rows.forEach((r) => {
      const tr = tb.insertRow();
      const idx = tr.insertCell(); idx.textContent = r.i; idx.className = "pr-idx";
      r.cells.forEach((v, j) => {
        const td = tr.insertCell(); const k = byCell.get(`${r.i}\u0000${cols[j]}`);
        td.textContent = fmt(v);
        if (k !== undefined) {
          const s = status[k];
          td.className = s === "accepted" ? "pr-accepted" : s === "dismissed" ? "pr-dismissed" : issues[k].kind !== "cell" ? "pr-label" : "pr-flag";
          if (s === "accepted") td.textContent = issues[k].kind === "cell" ? sig4(issues[k].suggested) : fmt(issues[k].suggested);
          if (k === sel) td.classList.add("pr-sel");
          td.title = `${issues[k].cause}`; td.dataset.issue = k;
          td.onclick = () => { model.set("selected", k); model.save_changes(); };
        }
      });
    });
    wrap.appendChild(table); body.appendChild(wrap);

    const side = document.createElement("div"); side.className = "pr-side";
    if (issues.length) {
      const r = issues[sel], s = status[sel];
      const range = r.kind === "cell" && r.low != null && r.high != null ? `<div class="pr-range">80% plausible: ${sig4(r.low)} – ${sig4(r.high)}</div>` : "";
      side.innerHTML = `
        <div class="pr-title">Issue ${sel + 1}/${issues.length} <span class="pr-st pr-st-${s}">${s}</span></div>
        <div>row <b>${r.row}</b> · <b>${r.column}</b></div>
        <div class="pr-kv"><span>recorded</span><b class="pr-red">${fmt(r.value)}</b></div>
        <div class="pr-kv"><span>TabPFN-3.5</span><b class="pr-green">${r.kind === "cell" ? sig4(r.suggested) : fmt(r.suggested)}</b></div>
        ${range}
        <div class="pr-kv"><span>surprise</span><b>${Number(r.surprise).toFixed(1)}</b></div>
        <div class="pr-cause">${r.cause}</div>
        ${r.evidence ? `<div class="pr-ev">related in this row: ${r.evidence}</div>` : ""}`;
      const btns = document.createElement("div"); btns.className = "pr-btns";
      [["◀", () => move(-1)], ["accept", () => setStatus(sel, "accepted")], ["dismiss", () => setStatus(sel, "dismissed")],
       ["undo", undo], ["▶", () => move(1)]].forEach(([t, f]) => {
        const b = document.createElement("button"); b.textContent = t; b.className = `pr-btn pr-btn-${t}`; b.onclick = f; btns.appendChild(b);
      });
      side.appendChild(btns);
    } else side.textContent = "No issues found.";
    (meta.patterns || []).forEach((p) => {
      const d = document.createElement("div"); d.className = "pr-pattern";
      d.textContent = `⚑ ${p.column}: ${p.value} in ${Math.round(p.share * 100)}% of rows (likely placeholder code)`; side.appendChild(d);
    });
    const note = document.createElement("div"); note.className = "pr-note";
    note.textContent = "Keys: n/p next/prev · a accept · d dismiss · u undo. A flag is a prompt to check the source, not proof of an error.";
    side.appendChild(note);
    body.appendChild(side); root.appendChild(body);
    const cur = root.querySelector(".pr-sel"); if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  root.addEventListener("keydown", (e) => {
    const sel = model.get("selected");
    if (e.key === "n") move(1); else if (e.key === "p") move(-1);
    else if (e.key === "a") setStatus(sel, "accepted"); else if (e.key === "d") setStatus(sel, "dismissed");
    else if (e.key === "u") undo(); else return;
    e.preventDefault();
  });
  ["change:rows", "change:status", "change:selected", "change:issues", "change:only_flagged"].forEach((ev) => model.on(ev, draw));
  draw();
  return () => ["change:rows", "change:status", "change:selected", "change:issues", "change:only_flagged"].forEach((ev) => model.off(ev, draw));
}

export default { render };
