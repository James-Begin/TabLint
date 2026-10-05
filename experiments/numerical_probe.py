"""Bounded, isolated TabPFN numerical probe; never changes the audit default.
Run serially with CUDA_VISIBLE_DEVICES=7, uv run python experiments/numerical_probe.py.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from tabpfn.architectures import tabpfn_v3_5 as v35
import tabpfn
import tabpfn.preprocessing.torch.ops as ops
import tabpfn.preprocessing.torch.torch_standard_scaler as scaler_module
import tabpfn.preprocessing.torch.torch_soft_clip_outliers as outlier_module
from auditkit.engine import AuditConfig, FeatureDomain, _TorchAdapter
from fz import data


class SafeSqrt(torch.autograd.Function):
    """Exact sqrt forward; bounded zero derivative only at variance == 0.

    This is an experimental backward convention, NOT robustness validation.
    Positive variances retain the ordinary derivative; no gradient masking.
    """
    @staticmethod
    def forward(ctx, variance):
        root = torch.sqrt(variance)
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, incoming):
        root, = ctx.saved_tensors
        positive = root > 0
        divisor = torch.where(positive, 2 * root, torch.ones_like(root))
        return incoming * torch.where(positive, 1 / divisor, torch.zeros_like(root))


def safe_nanstd(x, axis=0):
    # Same finite forward as installed ops.py:48-71; mask before square and
    # substitute the sqrt backward only at zero variance.
    nan_mask = torch.isnan(x)
    num_valid = torch.where(nan_mask, torch.zeros_like(x), torch.ones_like(x)).sum(dim=axis)
    value_sum = torch.where(nan_mask, torch.zeros_like(x), x).sum(dim=axis)
    mean = value_sum / num_valid.clamp(min=1.0)
    mean_broadcast = mean.unsqueeze(axis).expand_as(x)
    # Mask BEFORE square: square(NaN)'s backward otherwise receives 0 * NaN.
    difference = torch.where(nan_mask, torch.zeros_like(x), x - mean_broadcast)
    sq_diff = torch.square(difference).sum(dim=axis)
    variance = sq_diff / (num_valid - 1).clamp(min=1.0)
    return SafeSqrt.apply(variance)


@contextlib.contextmanager
def experimental_std_backward():
    original = scaler_module.torch_nanstd
    original_outlier = outlier_module.torch_nanstd
    scaler_module.torch_nanstd = safe_nanstd
    outlier_module.torch_nanstd = safe_nanstd
    try:
        yield
    finally:
        scaler_module.torch_nanstd = original
        outlier_module.torch_nanstd = original_outlier


def resources():
    commands = ["uptime", "free -h", "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv",
                "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv"]
    return {c: subprocess.run(c, shell=True, capture_output=True, text=True, check=True).stdout for c in commands}


def source_evidence():
    result = {}
    for obj in (ops.torch_nanstd, v35.TorchStandardScaler.fit, v35._ecdf_midrank_counts,
                v35._build_ecdf_context, v35._in_context_ecdf):
        lines, start = inspect.getsourcelines(obj)
        result[obj.__name__] = {"file": inspect.getfile(obj), "start": start,
                               "source": "".join(f"{start+i}: {line}" for i, line in enumerate(lines))}
    return result


def microscopic():
    # No model queries: isolate each suspect operation independently.
    patterns = {"constant": np.zeros(32), "binary": np.arange(32) % 2,
                "zero_heavy": np.r_[np.zeros(28), np.arange(1, 5)],
                "unique": np.linspace(-1, 1, 32)}
    result = []
    for name, values in patterns.items():
        for operation in ("scaler", "ecdf"):
            for patched in (False, True) if operation == "scaler" else (False,):
                x = torch.tensor(values.reshape(32, 1, 1), device="cuda", dtype=torch.float32, requires_grad=True)
                record = {"pattern": name, "operation": operation, "patched": patched}
                try:
                    with experimental_std_backward() if patched else contextlib.nullcontext():
                        with torch.autograd.detect_anomaly(check_nan=True):
                            if operation == "scaler":
                                scaler = v35.TorchStandardScaler()
                                output = scaler.transform(x, scaler.fit(x))
                            else:
                                bx = x.transpose(0, 1)
                                ecdf = v35._build_ecdf_context(bx, 32, 8192)
                                output = v35._in_context_ecdf(bx, ecdf)
                            grad, = torch.autograd.grad((output * torch.linspace(0.5, 1.5, 32, device="cuda")[:, None, None]).sum(), x)
                    record.update(finite=bool(torch.isfinite(grad).all()), grad_max=float(grad.abs().max()))
                except RuntimeError:
                    record.update(finite=False, traceback=traceback.format_exc())
                result.append(record)
    return result


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("Must expose ONLY physical GPU7")
    # Independent race-aware occupancy check before CUDA initialization.
    occupancy = subprocess.run("nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, check=True).stdout.strip()
    if any(int(v.strip()) > 0 for v in occupancy.split(",")):
        raise RuntimeError(f"GPU7 occupied; stop without waiting or killing: {occupancy}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    out = Path("results/numerical_probe")
    out.mkdir(parents=True, exist_ok=True)
    result = {"version": tabpfn.__version__, "before": resources(), "source": source_evidence(),
              "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "gpu_count": torch.cuda.device_count(), "physical_gpu": 7, "query_limit": 20,
              "rows_per_call_limit": 3000, "forward_tolerance": 1e-7, "datasets": []}
    original = scaler_module.torch_nanstd
    result["microscopic"] = microscopic()
    queries = 0
    inference_rows = 0
    def call(adapter, xc, yc, target, attack, label, *, grad=False, anomaly=False, patched=False):
        nonlocal queries, inference_rows
        if queries >= 20 or time.monotonic() - started > 850:
            raise RuntimeError("Probe budget exhausted")
        rows = len(xc) + len(attack) + len(target)
        assert rows <= 3000
        queries += 1
        inference_rows += rows
        record = {"query": queries, "rows": rows, "patched": patched, "gradient_requested": grad,
                  "anomaly": anomaly}
        raw_attack = torch.tensor(attack, dtype=torch.float32, device="cuda", requires_grad=grad)
        raw_context = torch.tensor(xc, dtype=torch.float32, device="cuda")
        raw_target = torch.tensor(target, dtype=torch.float32, device="cuda")
        labels = torch.tensor(np.r_[yc, label], dtype=torch.float32, device="cuda")
        try:
            with experimental_std_backward() if patched else contextlib.nullcontext():
                with torch.autograd.detect_anomaly(check_nan=True) if anomaly else contextlib.nullcontext():
                    with torch.enable_grad() if grad else torch.no_grad():
                        adapter._freeze()
                        probability = adapter.core.proba(adapter.clf,
                            adapter._standardize(torch.cat((raw_context, raw_attack))), labels,
                            adapter._standardize(raw_target))[0, 1]
                        adapter._freeze()
                        record["p"] = float(probability.detach().cpu())
                        if grad:
                            gradient, = torch.autograd.grad(probability, raw_attack)
                            g = gradient.detach().cpu().numpy().astype(float)
                            record.update(finite=bool(np.isfinite(g).all()), nonfinite=int((~np.isfinite(g)).sum()))
                            if record["finite"]:
                                record["gradient_raw"] = g.tolist()
        except RuntimeError:
            record["traceback"] = traceback.format_exc()
        assert scaler_module.torch_nanstd is original
        print(json.dumps(record), flush=True)
        return record

    for name in ("credit-g", "taiwan", "breast_cancer"):
        if name == "breast_cancer":
            bunch = load_breast_cancer()
            selection = np.random.default_rng(0).permutation(len(bunch.target))[:65]
            x, y = bunch.data[selection].astype(np.float32), bunch.target[selection]
            meta = {"names": list(bunch.feature_names), "groups": [], "numeric": np.arange(x.shape[1])}
        else:
            x, y, meta = data.load_meta(name, n_max=65, seed=0)
        context, yc, target = x[:63], y[:63], x[63:64]
        cfg = AuditConfig(seed=0, device="cuda", n_estimators=1,
                          groups=tuple(tuple(map(int, g)) for g in meta["groups"]))
        domain = FeatureDomain.fit(context, cfg, meta["names"])
        nearest = int(np.argmin(np.linalg.norm(domain.standardize(context) - domain.standardize(target), axis=1)))
        seed = context[nearest:nearest+1].copy()
        label = int(1-y[63])  # fixed deliberate corrupted target-class label, not a clean decision claim
        payload = {"context": context.tolist(), "y": yc.tolist(), "target": target.tolist(), "seed": seed.tolist(),
                   "seed_index": nearest, "label": label, "names": meta["names"], "groups": [list(map(int, g)) for g in meta["groups"]]}
        encoded = json.dumps(payload, sort_keys=True).encode()
        (out / f"{name}_rows.json").write_bytes(encoded)
        item = {"name": name, "rows_sha256": hashlib.sha256(encoded).hexdigest(),
                "constant_context_columns": np.flatnonzero(context.std(0) == 0).tolist(),
                "n_features": x.shape[1], "calls": []}
        adapter = _TorchAdapter(cfg, domain)
        # Initialize weights without a target query, then freeze before any forward.
        adapter.clf.fit_with_differentiable_input(
            torch.tensor(domain.standardize(context), dtype=torch.float32, device="cuda"),
            torch.tensor(yc, dtype=torch.float32, device="cuda"))
        adapter._freeze()
        item["frozen_parameter_elements"] = adapter.frozen_parameters
        # Two independent fits on bit-identical rows: anomaly pinpoints op, normal backward counts failures.
        first = call(adapter, context, yc, target, seed, label, grad=True, anomaly=True)
        repeat = call(adapter, context, yc, target, seed, label, grad=True)
        fixed = call(adapter, context, yc, target, seed, label, grad=True, patched=True, anomaly=True)
        item["calls"].extend([first, repeat, fixed])
        item["exact_repeat_p_difference"] = abs(first["p"] - repeat["p"])
        item["patched_forward_difference"] = abs(first["p"] - fixed["p"])
        item["forward_equivalent"] = item["patched_forward_difference"] <= result["forward_tolerance"]
        # Only compare progress after forward equivalence and finite gradient established.
        if name != "breast_cancer" and item["forward_equivalent"] and fixed.get("finite"):
            current = seed.copy()
            base = fixed
            history = []
            for step in range(2):
                g = (2*label-1) * np.asarray(base["gradient_raw"]) * domain.std
                continuous = current + 0.05 * np.sign(g) * domain.std
                candidate = domain.project(continuous, seed)
                check = domain.validate(candidate, target, seed)
                progress = call(adapter, context, yc, target, candidate, label, grad=True, patched=True)
                item["calls"].append(progress)
                history.append({"step": step+1, "valid": check, "objective_class": label,
                                "objective_delta": (2*label-1) * (progress["p"] - fixed["p"]),
                                "candidate": candidate.tolist()})
                if not progress.get("finite"):
                    break
                base = progress
                current = candidate
            item["progress"] = history
        if name == "breast_cancer" and fixed.get("finite"):
            # Put one continuous coordinate at midpoint of a nonempty empirical gap.
            j = int(meta["numeric"][0])
            unique = np.unique(context[:, j])
            gaps = np.diff(unique)
            gap_idx = int(np.argmax(gaps))
            center = seed.copy()
            center[0, j] = (float(unique[gap_idx]) + float(unique[gap_idx+1])) / 2
            h = float(gaps[gap_idx]) * 0.1
            middle = call(adapter, context, yc, target, center, label, grad=True, patched=False)
            item["calls"].append(middle)
            perturb = []
            for sign in (-1, 1):
                candidate = center.copy()
                candidate[0, j] += sign*h
                record = call(adapter, context, yc, target, candidate, label)
                perturb.append(record)
                item["calls"].append(record)
            fd = (perturb[1]["p"]-perturb[0]["p"]) / (2*h)
            item["finite_difference"] = {"feature": j, "center": float(center[0,j]), "h": h,
                "gap_edges": unique[gap_idx:gap_idx+2].tolist(), "central_derivative": fd,
                "autograd_raw": middle.get("gradient_raw", [[None]])[0][j] if middle.get("finite") else None,
                "rank_ties_crossed": False, "caveat": "One coordinate only; other channels and larger rank-crossing changes are not validated."}
        result["datasets"].append(item)
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        del adapter
        torch.cuda.empty_cache()
    result.update(queries=queries, total_inference_rows=inference_rows,
                  elapsed_seconds=time.monotonic()-started, monkeypatch_restored=scaler_module.torch_nanstd is original,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), after=resources(),
                  claim="Experimental gradient convention only; not validated robustness or attribution.")
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("queries", "total_inference_rows", "elapsed_seconds", "monkeypatch_restored", "gpu_peak_allocated_bytes")}), flush=True)


def followup():
    """At most four additional queries, on the development suite's one-hot schema.

    Requires the first run's ledger (16 calls). Uses only a <=100-row reduction,
    never claims it is the original 200-row reproduction.
    """
    from auditkit.cli import dataset
    out = Path("results/numerical_probe")
    ledger = json.loads((out / "summary.json").read_text())
    assert ledger["queries"] <= 16
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "7"
    occupancy = subprocess.run("nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, check=True).stdout.strip()
    if any(int(v.strip()) > 0 for v in occupancy.split(",")):
        raise RuntimeError(f"GPU7 occupied; stop: {occupancy}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    before = resources()
    start = time.monotonic()
    # Fixed pre-existing failing record, not selected by a new target query.
    report_path = Path("results/development/taiwan_102_1.json")
    report = json.loads(report_path.read_text())
    x, y, meta = dataset("taiwan")
    ids = report["experiment"]["context_ids"][:97]
    target_id = report["experiment"]["target_id"]
    context, yc, target = x[ids], y[ids], x[target_id:target_id+1]
    cfg = AuditConfig(seed=102, device="cuda", n_estimators=1,
                      groups=tuple(tuple(g) for g in meta["groups"]))
    domain = FeatureDomain.fit(context, cfg, meta["names"])
    nearest = np.argsort(np.linalg.norm(domain.standardize(context)-domain.standardize(target), axis=1), kind="stable")[:3]
    seeds = context[nearest].copy()
    objective = 1 - report["baseline"]["predicted_class"]
    labels = np.full(3, objective)
    adapter = _TorchAdapter(cfg, domain)
    adapter.clf.fit_with_differentiable_input(torch.tensor(domain.standardize(context), dtype=torch.float32, device="cuda"),
                                             torch.tensor(yc, dtype=torch.float32, device="cuda"))
    adapter._freeze()
    payload = {"context": context.tolist(), "y": yc.tolist(), "target": target.tolist(),
               "seeds": seeds.tolist(), "labels": labels.tolist(), "context_ids": ids,
               "target_id": target_id, "groups": meta["groups"], "names": meta["names"]}
    encoded = json.dumps(payload, sort_keys=True).encode()
    (out / "taiwan_onehot_rows.json").write_bytes(encoded)
    result = {"before": before, "calls": [], "rows_sha256": hashlib.sha256(encoded).hexdigest(),
              "development_report": str(report_path), "reduced_context_rows": 97, "attack_rows": 3,
              "frozen_parameter_elements": adapter.frozen_parameters, "objective_class": objective,
              "source": source_evidence(), "cpu_threads": 2, "gpu_count": torch.cuda.device_count()}
    lines, line_start = inspect.getsourcelines(outlier_module.TorchSoftClipOutliers.fit)
    result["outlier_fit_source"] = {"file": inspect.getfile(outlier_module), "source": "".join(f"{line_start+i}: {line}" for i,line in enumerate(lines))}
    original = scaler_module.torch_nanstd
    original_outlier = outlier_module.torch_nanstd
    def query(rows, patched, anomaly):
        assert len(result["calls"]) < 4
        assert time.monotonic()-start < 850
        record = {"query": ledger["queries"] + len(result["calls"]) + 1, "patched": patched, "anomaly": anomaly, "rows": 101}
        try:
            with experimental_std_backward() if patched else contextlib.nullcontext():
                with torch.autograd.detect_anomaly(check_nan=True) if anomaly else contextlib.nullcontext():
                    p, grad = adapter.value_and_grad(np.r_[domain.standardize(context), domain.standardize(rows)],
                                                   np.r_[yc, labels], domain.standardize(target), 3, objective)
                    record.update(p=p, finite=bool(np.isfinite(grad).all()), nonfinite=int((~np.isfinite(grad)).sum()))
                    if record["finite"]:
                        record["gradient_standardized"] = grad.tolist()
        except RuntimeError:
            record["traceback"] = traceback.format_exc()
            # anomaly may fail after forward, so retain probability with a hook (not another query).
        result["calls"].append(record)
        print(json.dumps(record), flush=True)
        assert scaler_module.torch_nanstd is original and outlier_module.torch_nanstd is original_outlier
        return record
    first = query(seeds, False, True)
    repeat = query(seeds, False, False)
    fixed = query(seeds, True, True)
    if "p" in repeat and "p" in fixed:
        result["patched_forward_difference"] = abs(repeat["p"]-fixed["p"])
        result["forward_equivalent"] = result["patched_forward_difference"] <= 1e-7
        if result["forward_equivalent"] and fixed.get("finite"):
            raw = seeds + 0.05*np.sign(np.asarray(fixed["gradient_standardized"]))*domain.std
            candidate = domain.project(raw, seeds)
            result["candidate_validity"] = domain.validate(candidate, target, seeds)
            progress = query(candidate, True, False)
            result["objective_delta"] = (2*objective-1)*(progress["p"]-fixed["p"])
            result["candidate"] = candidate.tolist()
    result.update(elapsed_seconds=time.monotonic()-start, after=resources(),
                  queries=len(result["calls"]), cumulative_queries=ledger["queries"]+len(result["calls"]),
                  total_inference_rows=ledger["total_inference_rows"]+101*len(result["calls"]),
                  monkeypatch_restored=scaler_module.torch_nanstd is original and outlier_module.torch_nanstd is original_outlier,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (out / "followup.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:result[k] for k in ("queries", "cumulative_queries", "total_inference_rows", "elapsed_seconds", "monkeypatch_restored")}), flush=True)


def ops_only():
    """No target queries: isolate clipping and inspect existing public-row evidence."""
    from auditkit.cli import dataset
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "7"
    occupancy = subprocess.run("nvidia-smi -i 7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, check=True).stdout.strip()
    if any(int(v.strip()) > 0 for v in occupancy.split(",")):
        raise RuntimeError(f"GPU7 occupied; stop: {occupancy}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    before = resources()
    result = {"target_queries": 0, "before": before, "tests": []}
    for sigma in (4.0, 12.0):
        for patched in (False, True):
            raw = torch.tensor(np.r_[np.zeros(31), 1].reshape(32, 1), dtype=torch.float32, device="cuda", requires_grad=True)
            record = {"n_sigma": sigma, "patched": patched, "rows": 32}
            with experimental_std_backward() if patched else contextlib.nullcontext():
                try:
                    with torch.autograd.detect_anomaly(check_nan=True):
                        clip = outlier_module.TorchSoftClipOutliers(n_sigma=sigma)
                        fitted = clip.fit(raw)
                        output = clip.transform(raw, fitted)
                        record.update(lower=fitted["lower"].detach().cpu().tolist(),
                                      upper=fitted["upper"].detach().cpu().tolist(),
                                      output=output.detach().cpu().tolist())
                        grad, = torch.autograd.grad(output.sum(), raw)
                        record.update(finite=bool(torch.isfinite(grad).all()), gradient=grad.detach().cpu().tolist())
                except RuntimeError:
                    record.update(finite=False, traceback=traceback.format_exc())
            result["tests"].append(record)
    result["forward_identical_sigma4"] = result["tests"][0]["output"] == result["tests"][1]["output"]
    result["forward_identical_sigma12"] = result["tests"][2]["output"] == result["tests"][3]["output"]
    # Explicit ECDF tie discontinuity without any model inference.
    bx = torch.tensor([0., 0., 1., 1.], device="cuda").reshape(1,4,1)
    buckets = v35._build_ecdf_context(bx, 4, 8192)
    q = torch.tensor([0.9999, 1., 1.0001], device="cuda", requires_grad=True)
    ranks = v35._in_context_ecdf(q.reshape(1,3,1), buckets)
    grad, = torch.autograd.grad(ranks.sum(), q)
    result["rank_tie"] = {"queries": q.detach().cpu().tolist(), "ranks": ranks.detach().cpu().flatten().tolist(),
                          "gradient": grad.cpu().tolist(), "central_fd_at_tie": float(((ranks.flatten()[2]-ranks.flatten()[0])/0.0002).detach())}
    # Inspection only of original 200-row public contexts; no >100-row autograd or model call.
    fullx, fully, meta = dataset("taiwan")
    report = json.loads(Path("results/development/taiwan_102_1.json").read_text())
    ids = report["experiment"]["context_ids"]
    x, y = fullx[ids], fully[ids]
    reconstructed_hash = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
    dom = FeatureDomain.fit(x, AuditConfig(groups=tuple(tuple(g) for g in meta["groups"])))
    # NumPy inspection of the two-pass clipping masks, after entry standardization.
    near = next(a for a in report["attacks"] if a["method"] == "near_rows")
    augmented = np.r_[x, np.asarray(near["rows"], dtype=np.float32)]
    z = dom.standardize(augmented).astype(np.float32)
    m, sd = z.mean(0), z.std(0, ddof=1)
    outliers = np.abs(z-m) > 12*sd
    clean = np.where(outliers, np.nan, z)
    clean_std = np.nanstd(clean, axis=0, ddof=1)
    suspect = np.flatnonzero((sd > 0) & (clean_std == 0))
    result["original_row_inspection"] = {"context_hash_matches": reconstructed_hash == report["experiment"]["context_sha256"],
        "context_sha256": reconstructed_hash, "inspection_rows": len(augmented), "autograd_rows": 0,
        "newly_constant_after_sigma12": suspect.tolist(),
        "columns_masked_as_nan": np.flatnonzero(outliers.any(0)).tolist(),
        "masked_names": [meta["names"][i] for i in np.flatnonzero(outliers.any(0))],
        "removed_counts": outliers[:,outliers.any(0)].sum(0).tolist(),
        "candidate_rows_source": "near_rows from fixed existing failing report; no new inference", 
        "reported_nonfinite": next(a["nonfinite_gradient_entries"] for a in report["attacks"] if a["method"] == "gradient"),
        "caveat": "Stats-only mechanism evidence; original full-model failing backward not rerun under <=100-row limit."}
    result["after"] = resources()
    result["monkeypatch_restored"] = scaler_module.torch_nanstd is ops.torch_nanstd and outlier_module.torch_nanstd is ops.torch_nanstd
    Path("results/numerical_probe/ops_only.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("before", "after", "tests")}), flush=True)


if __name__ == "__main__":
    import sys
    if "--ops-only" in sys.argv:
        ops_only()
    elif "--followup" in sys.argv:
        followup()
    else:
        main()
