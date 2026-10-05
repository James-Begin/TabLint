"""High-level investigation API over auditkit (suspect → prove → catch → report).

All numbers come from the underlying audit engine and refits; this module only
orchestrates and formats. Nothing here is a robustness certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from auditkit.engine import AuditConfig, FeatureDomain, audit

DEFAULT_RECEIVERS = ("tabpfn_standard", "hgb", "logistic")
THREAT_MODEL = ("Threat model: k rows are appended to the training context with adversarially chosen "
                "labels; feature values are schema-valid and bounded near real rows (geometric checks, "
                "not domain realism).")


@dataclass
class Finding:
    """Result of the gradient search on the differentiable surrogate (not yet verified)."""
    report: dict
    X: np.ndarray
    y: np.ndarray
    target: np.ndarray
    attack: dict

    @property
    def rows(self) -> np.ndarray:
        return np.asarray(self.attack["rows"], dtype=np.float32).reshape(-1, self.X.shape[1])

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.attack["labels"], dtype=int)

    @property
    def surrogate_flipped(self) -> bool:
        return bool(self.attack["flipped"])

    @property
    def p_before(self) -> float:
        return float(self.attack["p_before"])

    @property
    def p_after(self) -> float:
        return float(self.attack["p_after"])

    @property
    def valid(self) -> bool:
        return bool(self.attack["valid"])

    @property
    def trace(self) -> list:
        return self.attack.get("optimization_history", [])


@dataclass
class Proof:
    """Refit verification of the exact appended rows on named receiver models."""
    rows: list = field(default_factory=list)
    controls: list = field(default_factory=list)

    def flipped(self, model: str) -> bool:
        return any(r["model"] == model and r.get("flipped") for r in self.rows)

    @property
    def verified(self) -> bool:
        """True iff the deployed model (standard TabPFN-3.5) flips and the benign-label control does not."""
        return any(r["model"] == "tabpfn_standard" and r.get("flipped") and not r.get("benign_label_flip")
                   for r in self.rows)


@dataclass
class Suspects:
    """Exact leave-one-out ranking of every row of the (poisoned) context."""
    ranking: list  # [{row, delta_p, label, planted}]
    p_before: float
    planted_in_top_k: int | None
    k: int | None
    caveat: str = ""


class Case:
    """A dataset + model configuration under investigation."""

    def __init__(self, X, y, feature_names: Sequence[str] | None = None, *, device: str = "cpu",
                 groups: Sequence[Sequence[int]] = (), integer_indices: Sequence[int] = (),
                 protected_indices: Sequence[int] = (), seed: int = 0):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y).astype(int)
        self.names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(self.X.shape[1])]
        self.device, self.seed = device, seed
        self._kw = dict(groups=tuple(tuple(g) for g in groups), integer_indices=tuple(integer_indices),
                        protected_indices=tuple(protected_indices))

    # 1 ------------------------------------------------------------------
    def suspect(self, target, *, k: int = 3, steps: int = 12, lr: float = 0.08,
                min_radius: float = 0.5, max_radius: float = 1.5, stable_backward: bool = True) -> Finding:
        """Gradient search through TabPFN's in-context learning for k appended rows that flip ``target``."""
        target = np.asarray(target, dtype=np.float32).reshape(1, -1)
        config = AuditConfig(seed=self.seed, steps=steps, lr=lr, k=k, device=self.device,
                             stable_backward=stable_backward, **self._kw)
        domain = FeatureDomain.fit(self.X, config, self.names)
        config = replace(config, min_target_distance=min_radius * domain.nn_radius,
                         max_real_distance=max_radius * domain.nn_radius if max_radius else None)
        report = audit(self.X, self.y, target, self.names, config)
        report["feature_names"], report["target"] = self.names, target[0].tolist()
        attack = next(a for a in report["attacks"] if a["method"] == "gradient")
        return Finding(report, self.X, self.y, target, attack)

    # 2 ------------------------------------------------------------------
    def prove(self, finding: Finding, receivers: Sequence[str] = DEFAULT_RECEIVERS) -> Proof:
        """Refit each receiver on context + the exact rows/labels; also run the benign-label control."""
        from experiments.verify import verify_report
        narrowed = dict(finding.report, attacks=[finding.attack])
        rows, controls = verify_report(narrowed, finding.X, finding.y, finding.target, list(receivers),
                                       self.seed, self.device)
        return Proof(rows, controls)

    # 3 ------------------------------------------------------------------
    def catch(self, finding: Finding | None = None, *, X=None, y=None, target=None,
              limit: int | None = None) -> Suspects:
        """Rank every context row by |change in P(target)| when removed (exact refits).

        With a ``finding``, the appended rows are included and ``planted_in_top_k`` counts how many of
        them land in the top-k. Without one, pass ``X``, ``y`` and ``target`` to audit any context.
        """
        from auditkit.influence import leave_one_out
        if finding is not None:
            X = np.concatenate([finding.X, finding.rows])
            y = np.concatenate([finding.y, finding.labels])
            target = finding.target
            planted = set(range(len(finding.X), len(X)))
        else:
            if X is None or y is None or target is None:
                raise ValueError("provide a finding or X, y and target")
            planted = set()
        loo = leave_one_out(X, y, target, device=self.device, seed=self.seed, limit=limit)
        ranking = sorted(({"row": r["row"], "delta_p": abs(r["delta_p_positive"]),
                           "signed_delta_p": r["delta_p_positive"], "label": r["training_label"],
                           "planted": r["row"] in planted}
                          for r in loo["rows"] if "delta_p_positive" in r), key=lambda r: -r["delta_p"])
        k = len(planted) or None
        hits = sum(r["planted"] for r in ranking[:k]) if k else None
        return Suspects(ranking, loo["p_before"], hits, k, loo["caveat"])

    # report -------------------------------------------------------------
    def report(self, finding: Finding, proof: Proof | None = None, suspects: Suspects | None = None) -> str:
        f = finding
        pred = lambda p: int(p >= 0.5)
        out = ["# Chain of Custody — investigation report", "", THREAT_MODEL, "",
               "## 1. Suspect (differentiable surrogate; not yet verified)",
               f"- Query decision on surrogate: P(class 1) = {f.p_before:.4f} → class {pred(f.p_before)}",
               f"- After appending {len(f.rows)} rows: P = {f.p_after:.4f} → class {pred(f.p_after)}; "
               f"flip={f.surrogate_flipped}, constraints valid={f.valid}",
               f"- Appended labels: {f.labels.tolist()}",
               "", "| Feature | Query | " + " | ".join(f"Row {i+1}" for i in range(len(f.rows))) + " |",
               "|---|---:|" + "---:|" * len(f.rows)]
        # Show the features where appended rows differ most from the query.
        spread = np.abs(f.rows - f.target).max(axis=0) / (f.X.std(axis=0) + 1e-9)
        for j in np.argsort(-spread)[:8]:
            out.append(f"| {self.names[j]} | {f.target[0, j]:.4g} | " + " | ".join(f"{v:.4g}" for v in f.rows[:, j]) + " |")
        if proof is not None:
            out += ["", "## 2. Prove (refit on the exact rows and labels)",
                    "| Receiver | P before | P after | Decision flipped | Benign-label control flipped |", "|---|---:|---:|---|---|"]
            for r in proof.rows:
                if "p_after" in r:
                    out.append(f"| {r['model']} | {r['p_before']:.4f} | {r['p_after']:.4f} | {r['flipped']} | {r['benign_label_flip']} |")
            out.append(f"\n**Verified on deployed TabPFN-3.5 (flip and benign-label control clean): {proof.verified}**")
        if suspects is not None:
            out += ["", "## 3. Catch (exact leave-one-out, context + appended rows)"]
            if suspects.k:
                out.append(f"- Appended rows ranked in the top {suspects.k}: **{suspects.planted_in_top_k}/{suspects.k}**")
            out += ["", "| Rank | Row | |Δp| | Label | Appended |", "|---:|---:|---:|---:|---|"]
            for i, r in enumerate(suspects.ranking[:8], 1):
                out.append(f"| {i} | {r['row']} | {r['delta_p']:.4f} | {r['label']} | {r['planted']} |")
            out.append(f"\n{suspects.caveat}")
        out += ["", "## Limits",
                "- A single finite search is not a robustness certificate or a minimum-budget estimate.",
                "- Surrogate (1 estimator, identity preprocessing) differs from the deployed estimator; "
                "only the Prove section describes deployed behaviour.",
                "- Row influence is pipeline sensitivity, not causal responsibility; no fairness, medical or credit conclusions."]
        return "\n".join(out)

    def save(self, path, finding: Finding, proof: Proof | None = None, suspects: Suspects | None = None):
        """Write the self-contained JSON evidence (same schema the replay app reads)."""
        import json
        from pathlib import Path
        rep = dict(finding.report)
        if proof is not None:
            rep["verification"], rep["receiver_controls"] = proof.rows, proof.controls
        if suspects is not None:
            rep["custody_suspects"] = {"ranking": suspects.ranking[:25], "planted_in_top_k": suspects.planted_in_top_k, "k": suspects.k}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(rep, indent=2, allow_nan=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        return path
