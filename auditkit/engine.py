"""Single-target, serial poisoning audit of a differentiable surrogate.

Pure domain helpers require only NumPy. Model dependencies are imported on the
first model call. Distances and all metadata are fitted on the supplied context,
never on the target or a held-out dataset. Validity is geometric, not a claim of
medical/credit plausibility, and an unsuccessful search is not a certificate.

An optional adapter implements probability(X_standardized, y, target_standardized)
returning P(class=1), and value_and_grad(X, y, target, n_attack, objective_class)
returning (P(class=1), gradient of P(objective_class) with respect to the last
n_attack standardized rows). Adapters are query-local and called serially.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import time
from typing import Any

import numpy as np

GRAD_MODEL = "tabpfn_3_5_differentiable_surrogate"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class AuditConfig:
    seed: int = 0
    steps: int = 30
    lr: float = 0.08
    k: int = 3
    n_estimators: int = 1
    device: str = "cpu"
    protected_indices: tuple = ()
    groups: tuple = ()
    integer_indices: tuple = ()
    min_target_distance: float = 0.0
    max_real_distance: float | None = None
    stable_backward: bool = False  # opt-in, see auditkit/stability.py

    def __post_init__(self):
        for name, minimum in (("seed", 0), ("steps", 0), ("k", 1), ("n_estimators", 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        for name in ("lr", "min_target_distance", "max_real_distance"):
            value = getattr(self, name)
            if value is None and name == "max_real_distance":
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{name} must be finite numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not isinstance(self.stable_backward, (bool, np.bool_)):
            raise ValueError("stable_backward must be boolean")
        object.__setattr__(self, "stable_backward", bool(self.stable_backward))
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|mps|cuda(?::\d+)?", self.device):
            raise ValueError("device must be cpu, mps, or cuda[:index]")
        for name in ("protected_indices", "integer_indices"):
            values = tuple(_integer(i, name) for i in getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name}")
            object.__setattr__(self, name, values)
        groups = tuple(tuple(_integer(i, "group index") for i in g) for g in self.groups)
        flat = [i for g in groups for i in g]
        if any(not g for g in groups) or len(set(flat)) != len(flat):
            raise ValueError("category groups must be nonempty and disjoint")
        object.__setattr__(self, "groups", groups)


def _array(value, name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite real numeric data")
    with np.errstate(over="ignore", invalid="ignore"):
        a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} is not representable as finite float32")
    return a


def _nearest(z: np.ndarray, context: np.ndarray, self_rows: bool = False) -> np.ndarray:
    result = np.empty(len(z), dtype=np.float64)
    # Bound temporary allocation, including for large contexts/features.
    for i in range(len(z)):
        best = math.inf
        for start in range(0, len(context), 512):
            dist = np.linalg.norm(context[start:start + 512] - z[i], axis=1)
            if self_rows and start <= i < start + len(dist):
                dist[i - start] = np.inf
            best = min(best, float(dist.min()))
        result[i] = best
    return result


@dataclass(frozen=True)
class FeatureDomain:
    context: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nn_radius: float
    config: AuditConfig
    feature_names: tuple

    @classmethod
    def fit(cls, X_context, config: AuditConfig | None = None, feature_names=None):
        config = AuditConfig() if config is None else config
        if not isinstance(config, AuditConfig):
            raise ValueError("config must be AuditConfig")
        x = _array(X_context, "X_context")
        if x.ndim != 2 or len(x) < 2 or x.shape[1] == 0:
            raise ValueError("X_context must have at least two rows and one feature")
        d = x.shape[1]
        indices = config.protected_indices + config.integer_indices + tuple(i for g in config.groups for i in g)
        if any(i >= d for i in indices):
            raise ValueError("feature index out of range")
        for g in config.groups:
            values = x[:, g]
            if not ((values == 0) | (values == 1)).all() or not (values.sum(1) == 1).all():
                raise ValueError("context category groups must be hard one-hot")
        if config.integer_indices and not (x[:, config.integer_indices] == np.rint(x[:, config.integer_indices])).all():
            raise ValueError("declared integer features must be integer in context")
        names = tuple(feature_names) if feature_names is not None else tuple(f"x{i}" for i in range(d))
        if len(names) != d or any(not isinstance(n, str) for n in names):
            raise ValueError("feature_names must contain one string per feature")
        x = x.copy()
        mean, std = x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)
        std = np.where(std == 0, 1.0, std)
        lower, upper = x.min(0), x.max(0)
        radius = float(np.median(_nearest((x - mean) / std, (x - mean) / std, True)))
        for a in (x, mean, std, lower, upper):
            a.setflags(write=False)
        return cls(x, mean, std, lower, upper, radius, config, names)

    def standardize(self, rows):
        return (np.asarray(rows, dtype=np.float64) - self.mean) / self.std

    def project(self, rows, seed_rows):
        rows = _array(rows, "rows")
        seeds = _array(seed_rows, "seed_rows")
        if rows.ndim != 2 or rows.shape != seeds.shape or rows.shape[1] != self.context.shape[1]:
            raise ValueError("rows and seed_rows must have matching (n,d) shape")
        out = np.clip(rows, self.lower, self.upper).copy()
        ints = self.config.integer_indices
        if ints:
            out[:, ints] = np.clip(np.rint(out[:, ints]), np.ceil(self.lower[list(ints)]), np.floor(self.upper[list(ints)]))
        protected = set(self.config.protected_indices)
        if protected:
            pidx = list(protected)
            if not ((seeds[:, pidx] >= self.lower[pidx]) & (seeds[:, pidx] <= self.upper[pidx])).all():
                raise ValueError("protected seed values outside context bounds cannot be projected")
            pi = list(protected.intersection(ints))
            if pi and not (seeds[:, pi] == np.rint(seeds[:, pi])).all():
                raise ValueError("protected integer seed values must be integers")
        for g in self.config.groups:
            if not (((seeds[:, g] == 0) | (seeds[:, g] == 1)).all() and (seeds[:, g].sum(1) == 1).all()):
                raise ValueError("seed category groups must be hard one-hot")
            for r in range(len(out)):
                locked_hot = [i for i in g if i in protected and seeds[r, i] == 1]
                allowed = [i for i in g if self.upper[i] == 1 and (i not in protected or seeds[r, i] == 1)]
                choices = locked_hot or allowed
                if not choices:
                    raise ValueError("seed category cannot be preserved within context levels")
                hot = choices[int(np.argmax(out[r, choices]))]
                out[r, list(g)] = 0
                out[r, hot] = 1
        if protected:
            out[:, list(protected)] = seeds[:, list(protected)]
        return out.astype(np.float32)

    def validate(self, rows, target, seed_rows):
        return candidate_metrics(self, rows, target, seed_rows)

    def metadata(self):
        return {"feature_names": list(self.feature_names), "mean": self.mean.tolist(),
                "std": self.std.tolist(), "lower": self.lower.tolist(), "upper": self.upper.tolist(),
                "nn_radius": self.nn_radius, "fitted_on": "context_only"}


def project(domain: FeatureDomain, rows, seed_rows):
    """Hard projection; distance constraints are checked, not guaranteed by projection."""
    return domain.project(rows, seed_rows)


def candidate_metrics(domain: FeatureDomain, rows, target, seed_rows) -> dict:
    """Validate hard candidates; all rows must pass, and zero rows is invalid."""
    rows = _array(rows, "rows")
    seeds = _array(seed_rows, "seed_rows")
    target = _array(target, "target").reshape(-1)
    d = domain.context.shape[1]
    if rows.ndim != 2 or seeds.shape != rows.shape or rows.shape[1] != d or target.shape != (d,):
        raise ValueError("candidate, seed, or target shape mismatch")
    category_valid = all(bool((((rows[:, g] == 0) | (rows[:, g] == 1)).all()) and
                              (rows[:, g].sum(1) == 1).all()) for g in domain.config.groups)
    in_bounds = bool(((rows >= domain.lower) & (rows <= domain.upper)).all())
    integer_valid = bool((rows[:, domain.config.integer_indices] == np.rint(rows[:, domain.config.integer_indices])).all())
    protected_unchanged = bool((rows[:, domain.config.protected_indices] == seeds[:, domain.config.protected_indices]).all())
    dt = np.linalg.norm((rows.astype(np.float64) - target) / domain.std, axis=1)
    dn = _nearest(domain.standardize(rows), domain.standardize(domain.context))
    dt_min = float(dt.min()) if len(rows) else None
    dn_max = float(dn.max()) if len(rows) else None
    far = bool(len(rows) and dt_min >= domain.config.min_target_distance)
    near = bool(len(rows) and (domain.config.max_real_distance is None or dn_max <= domain.config.max_real_distance))
    return {"category_valid": category_valid, "in_bounds": in_bounds,
            "protected_unchanged": protected_unchanged, "integer_valid": integer_valid,
            "dtarget_min": dt_min, "dnn_max": dn_max, "min_target_distance_met": far,
            "max_real_distance_met": near,
            "valid": bool(len(rows) and category_valid and in_bounds and integer_valid and protected_unchanged and far and near)}


def validate_candidate(domain: FeatureDomain, rows, target, seed_rows) -> dict:
    return candidate_metrics(domain, rows, target, seed_rows)


class _TorchAdapter:
    def __init__(self, config, domain):
        import torch
        from fz import core
        from .stability import stable_backward
        self.torch, self.core, self.device = torch, core, config.device
        self.stable = lambda: stable_backward(config.stable_backward)
        self.clf = core.make_diff_clf(n_classes=2, n_estimators=config.n_estimators,
                                      seed=config.seed, device=config.device)
        self.frozen_parameters = 0
        self.domain = domain
        stats_dtype = torch.float32 if config.device == "mps" else torch.float64
        self.mean = torch.tensor(domain.mean, dtype=stats_dtype, device=self.device)
        self.std = torch.tensor(domain.std, dtype=stats_dtype, device=self.device)

    def _raw_tensor(self, standardized, requires_grad=False):
        # The public adapter protocol uses standardized coordinates; reconstruct
        # float32 raw inputs so the actual candidate graph contains the fixed,
        # context-only standardization before core.proba.
        raw = standardized * self.domain.std + self.domain.mean
        return self.torch.tensor(raw, dtype=self.torch.float32, device=self.device,
                                 requires_grad=requires_grad)

    def _standardize(self, raw):
        return ((raw.to(self.mean.dtype) - self.mean) / self.std).to(self.torch.float32)

    def _freeze(self):
        # TabPFN initializes model_ / models_ lazily. Avoid traversing arbitrary
        # caches or optimizer state, and only recurse through model containers.
        torch = self.torch
        seen = set()
        def visit(obj):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                for p in obj.parameters():
                    if p.requires_grad:
                        p.requires_grad_(False)
                        p.grad = None
                        self.frozen_parameters += p.numel()
            elif isinstance(obj, dict):
                for v in obj.values():
                    visit(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    visit(v)
        for attr in ("model_", "models_", "model", "models"):
            visit(getattr(self.clf, attr, None))

    def probability(self, X, y, target):
        t = self.torch
        with t.no_grad(), self.stable():
            p = self.core.proba(self.clf, self._standardize(self._raw_tensor(X)),
                                t.tensor(y, dtype=t.float32, device=self.device),
                                self._standardize(self._raw_tensor(target)))[0, 1]
        self._freeze()
        return float(p.detach().cpu())

    def value_and_grad(self, X, y, target, n_attack, objective_class):
        t = self.torch
        attack = self._raw_tensor(X[-n_attack:], requires_grad=True)
        context = self._raw_tensor(X[:-n_attack])
        self._freeze()
        with self.stable():
            p = self.core.proba(self.clf, self._standardize(t.cat((context, attack))),
                                t.tensor(y, dtype=t.float32, device=self.device),
                                self._standardize(self._raw_tensor(target)))[0, 1]
            self._freeze()
            objective = p if objective_class == 1 else 1 - p
            # Only attack-input gradients are requested; model parameter .grad is
            # never accumulated even if an unrecognized model container is present.
            grad, = t.autograd.grad(objective, attack)
        # Adapter contract is dP/dZ; autograd above computed dP/d(raw rows).
        return float(p.detach().cpu()), grad.detach().cpu().numpy().astype(np.float64) * self.domain.std


def _probability(value):
    p = float(value)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("model probability must be finite and in [0,1]")
    return p


def audit(X_context, y_context, target, feature_names=None, config=None, *, adapter=None) -> dict:
    """Audit one query. p_before/p_after always mean P(class=1).

    ``flipped`` is true only for a valid hard candidate with a changed decision.
    Unconstrained references additionally expose ``prediction_changed``. No
    attack outcome is a robustness certificate or a minimum-budget estimate.
    """
    config = AuditConfig() if config is None else config
    domain = FeatureDomain.fit(X_context, config, feature_names)
    x = domain.context
    original_y = np.asarray(y_context)
    y = _array(original_y.astype(np.int64) if original_y.dtype.kind == "b" else original_y, "y_context")
    if y.shape != (len(x),) or set(original_y.tolist()) != {0, 1}:
        raise ValueError("y_context must be one-dimensional binary labels containing both classes")
    y = y.astype(np.int64)
    target = _array(target, "target")
    if target.shape not in ((x.shape[1],), (1, x.shape[1])):
        raise ValueError("target must have shape (d,) or (1,d)")
    target = target.reshape(1, -1)
    model = adapter if adapter is not None else _TorchAdapter(config, domain)
    xc, xt = domain.standardize(x), domain.standardize(target)
    warnings = ["Geometric constraints do not establish medical or credit domain plausibility.",
                "Search failure or invalid attacks are not a robustness certificate; no minimum poison budget is claimed.",
                "Training labels on appended rows may be deliberately corrupted; context labels are unchanged.",
                "Only the differentiable surrogate is audited; deployed-model verification is out of scope."]
    p_before = _probability(model.probability(xc.copy(), y.copy(), xt.copy()))
    repeat = None
    try:
        repeat = _probability(model.probability(xc.copy(), y.copy(), xt.copy()))
    except Exception as exc:
        warnings.append(f"Identical-context repeatability check failed: {type(exc).__name__}.")
    repeatable = None if repeat is None else bool(abs(repeat - p_before) <= 1e-6)
    if repeatable is False:
        warnings.append("Identical-context predictions are not repeatable within 1e-6; attack effects may include model variability.")
    clean_class = int(p_before >= 0.5)
    attack_class = 1 - clean_class
    rng = np.random.default_rng(config.seed)
    attacks = []

    def evaluate(rows, labels):
        return _probability(model.probability(np.concatenate((xc, domain.standardize(rows))),
                                             np.concatenate((y, labels)), xt.copy()))

    def record(method, rows, labels, seeds, probability, started, evaluations=1, history=None, **extra):
        metrics = candidate_metrics(domain, rows, target, seeds)
        changed = bool(int(probability >= 0.5) != clean_class) if probability is not None else False
        valid = bool(metrics["valid"] and probability is not None)
        rec = {"method": method, "n_rows": int(len(rows)), "rows": rows.tolist(),
               "labels": labels.astype(int).tolist(), "p_before": p_before, "p_after": probability,
               "flipped": bool(valid and changed), "valid": valid, "constraints": metrics,
               "optimization_history": history or [], "evaluations": int(evaluations),
               "elapsed_s": float(time.perf_counter() - started), "masked_gradient_entries": 0,
               "prediction_changed": changed, **extra}
        attacks.append(rec)
        return rec

    start = time.perf_counter()
    exact = np.repeat(target, config.k, axis=0)
    labels = np.full(config.k, attack_class, dtype=np.int64)
    record("exact_duplicates", exact, labels, exact, evaluate(exact, labels), start,
           reference=True, constraints_enforced=False, unconstrained=True)

    start = time.perf_counter()
    selected = rng.choice(len(x), config.k, replace=config.k > len(x))
    real = x[selected].copy()
    real_labels = 1 - y[selected]
    record("random_real", real, real_labels, real, evaluate(real, real_labels), start,
           reference=True, constraints_enforced=False, source_indices=selected.tolist(),
           source_labels=y[selected].tolist(), labels_corrupted=True)

    start = time.perf_counter()
    nearest = np.argsort(np.linalg.norm(xc - xt, axis=1), kind="stable")
    seed_indices = np.resize(nearest, config.k)
    seeds = x[seed_indices].copy()
    # Start near actual context rows, preserving each seed's immutable values.
    # The finite trial set may be infeasible; report that instead of success.
    trials = [seeds]
    for scale in (0.05, 0.2, 0.5, 1.0):
        noisy = seeds.astype(np.float64) + rng.normal(size=seeds.shape) * domain.std * scale
        trials.append(domain.project(noisy, seeds))
    init = trials[0].copy()
    for trial in trials:
        if candidate_metrics(domain, trial, target, seeds)["valid"]:
            init = trial.copy()
            break
    near_p = evaluate(init, labels)
    near_rec = record("near_rows", init, labels, seeds, near_p, start,
                      reference=False, constraints_enforced=True, seed_indices=seed_indices.tolist())
    if not near_rec["valid"]:
        warnings.append("near_rows: no valid initialization found in the finite trial set.")

    start = time.perf_counter()
    # Step 0 is the EXACT evaluated near_rows, not a sigmoid/logit approximation.
    current = init.copy()
    best, best_p, best_valid = init.copy(), near_p, near_rec["valid"]
    history = [{"step": 0, "p_after": near_p, "valid": best_valid, "incumbent": best_valid}]
    evaluations = 0  # step 0 reuses the near_rows evaluation, explicitly disclosed
    valid_evaluations = int(best_valid)
    aborted = False
    nonfinite_entries = 0
    completed_steps = 0
    adam_m = np.zeros_like(current, dtype=np.float64)
    adam_v = np.zeros_like(current, dtype=np.float64)
    for step in range(1, config.steps + 1):
        try:
            evaluations += 1
            p_soft, grad = model.value_and_grad(
                np.concatenate((xc, domain.standardize(current))), np.concatenate((y, labels)),
                xt.copy(), config.k, attack_class)
            _probability(p_soft)
            grad = np.asarray(grad, dtype=np.float64)
            if grad.shape != current.shape:
                raise ValueError("adapter gradient shape mismatch")
            nonfinite_entries = int((~np.isfinite(grad)).sum())
            if nonfinite_entries:
                warnings.append(f"gradient: aborted on {nonfinite_entries} nonfinite gradient entries; none were masked, incumbent preserved.")
                aborted = True
                break
            # Adapter derivatives are in standardized coordinates. Updating in
            # those coordinates scales raw changes by context-only std.
            # Adam in standardized feature coordinates; otherwise small raw
            # probability derivatives make the search effectively stationary.
            adam_m = 0.9 * adam_m + 0.1 * grad
            adam_v = 0.999 * adam_v + 0.001 * np.square(grad)
            direction = (adam_m / (1 - 0.9 ** step)) / (
                np.sqrt(adam_v / (1 - 0.999 ** step)) + 1e-8)
            raw = current.astype(np.float64) + config.lr * direction * domain.std
            # Keep a continuous iterate so integer rounding and one-hot snapping
            # do not erase every small update. Only HARD projected rows are
            # evaluated as candidates; soft groups are an optimization device.
            raw = np.clip(raw, domain.lower, domain.upper)
            protected = list(config.protected_indices)
            if protected:
                raw[:, protected] = seeds[:, protected]
            for group in config.groups:
                locked = [i for i in group if i in config.protected_indices]
                free = [i for i in group if i not in locked and domain.upper[i] > 0]
                for row in range(config.k):
                    remainder = 1.0 - float(seeds[row, locked].sum()) if locked else 1.0
                    if free:
                        weights = np.maximum(raw[row, free], 0.0)
                        total = float(weights.sum())
                        raw[row, free] = remainder * (weights / total if total > 0 else np.ones(len(free)) / len(free))
            candidate = domain.project(raw, seeds)
            evaluations += 1
            p_candidate = evaluate(candidate, labels)
        except Exception as exc:
            warnings.append(f"gradient: aborted ({type(exc).__name__}: {exc}); incumbent preserved.")
            aborted = True
            break
        metrics = candidate_metrics(domain, candidate, target, seeds)
        valid = metrics["valid"]
        valid_evaluations += int(valid)
        score = p_candidate if attack_class == 1 else 1 - p_candidate
        best_score = best_p if attack_class == 1 else 1 - best_p
        improved = bool(valid and (not best_valid or score > best_score))
        if improved:
            best, best_p, best_valid = candidate.copy(), p_candidate, True
        history.append({"step": step, "p_after": p_candidate, "valid": valid, "incumbent": improved})
        current = raw
        completed_steps = step
    if not best_valid:
        warnings.append("gradient: zero valid evaluated candidates; result is invalid, not a successful attack or robustness certificate.")
    record("gradient", best, labels, seeds, best_p, start, evaluations, history,
           reference=False, constraints_enforced=True, seed_indices=seed_indices.tolist(),
           grad_model=GRAD_MODEL, aborted=aborted, nonfinite_gradient_entries=nonfinite_entries,
           completed_steps=completed_steps, valid_evaluations=valid_evaluations,
           initialization_evaluation_reused=True)
    return {"schema_version": "1.0", "config": asdict(config),
            "baseline": {"model": GRAD_MODEL, "p_positive": p_before, "predicted_class": clean_class,
                         "p_before": p_before, "p_class1": p_before, "prediction": clean_class,
                         "repeat_p_class1": repeat, "repeatable": repeatable, "repeatability_tolerance": 1e-6,
                         "grad_model": GRAD_MODEL},
            "attacks": attacks, "warnings": warnings,
            "diagnostics": {"grad_model": GRAD_MODEL, "adapter_injected": adapter is not None,
                            "domain": domain.metadata(), "target": target[0].tolist(),
                            "context_rows": int(len(x)), "decision_rule": "class1 iff p_class1 >= 0.5",
                            "parameter_elements_frozen": int(getattr(model, "frozen_parameters", 0)),
                            "gradient_aborted": aborted,
                            "stable_backward": config.stable_backward, "gradient_steps_completed": completed_steps,
                            "robustness_certificate": False, "minimum_budget_established": False}}
