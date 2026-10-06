"""Resource-bounded CPU capability experiment on text and retail identifiers.

Run ``python -m benchmarks.retail_cpu --help``. This directly tests TabPFN,
not the current TabLint context selector. No GPU or inference API is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CONFIG_PATH = Path(__file__).with_name("retail_config.json")
CONFIG = json.loads(CONFIG_PATH.read_text())
COMPACT_PATH = Path(__file__).with_name("retail_compact_cpu.json")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / "source.zip"
    if not source.exists():
        urllib.request.urlretrieve(CONFIG["source_url"], source)
    if sha256(source) != CONFIG["source_sha256"]:
        raise ValueError("Dataset checksum does not match the frozen configuration")
    xlsx = cache / "online_retail_II.xlsx"
    if not xlsx.exists():
        with zipfile.ZipFile(source) as z:
            with z.open("online_retail_II.xlsx") as src, xlsx.open("wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
    parquet = cache / "transactions.parquet"
    if not parquet.exists():
        print("Reading both original Excel worksheets…", flush=True)
        df = pd.concat(pd.read_excel(xlsx, sheet_name=None, engine="openpyxl").values(), ignore_index=True)
        df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})
        # Keep identifiers categorical; only Description is a text feature.
        for col in ["InvoiceNo", "StockCode", "CustomerID", "Country"]:
            df[col] = df[col].astype("string")
        df["Description"] = df.Description.astype("string").str.strip()
        missing = df[["Description", "CustomerID", "InvoiceDate", "UnitPrice", "Quantity"]].isna().any(axis=1)
        valid = (~missing & ~df.InvoiceNo.str.startswith("C", na=False)
                 & (df.Quantity > 0) & (df.UnitPrice > 0) & (df.Description.str.len() > 0))
        rows_source = len(df)
        rows_valid = int(valid.sum())
        df = df.loc[valid].drop_duplicates().sort_values(["InvoiceDate", "InvoiceNo"], kind="stable").reset_index(drop=True)
        df.to_parquet(parquet, index=False)
        write_json(cache / "cleaning.json", {"source_rows": rows_source, "valid_before_deduplication": rows_valid,
                  "retained_rows": len(df), "exact_duplicates_removed": rows_valid - len(df)})
    else:
        df = pd.read_parquet(parquet)
    tr, te = temporal_split(df)
    profile = {
        **json.loads((cache / "cleaning.json").read_text()),
        "source_sha256": sha256(source), "parquet_sha256": sha256(parquet),
        "source_url": CONFIG["source_url"], "license": "CC BY 4.0", "creator": "Daqing Chen",
        "cutoff": CONFIG["cutoff"], "training_pool_rows": len(tr), "future_pool_rows": len(te),
        "unique_stock_codes": int(df.StockCode.nunique()), "unique_customers": int(df.CustomerID.nunique()),
        "unique_descriptions": int(df.Description.nunique()),
        "description_length_median": float(df.Description.str.len().median()),
        "date_min": str(df.InvoiceDate.min()), "date_max": str(df.InvoiceDate.max()),
        "excluded_cross_boundary_invoices": int(df.InvoiceNo.nunique() - tr.InvoiceNo.nunique() - te.InvoiceNo.nunique()),
        "config_sha256": sha256(CONFIG_PATH),
    }
    write_json(out / "dataset_profile.json", profile)
    print(json.dumps(profile, indent=2), flush=True)


def temporal_split(df):
    cutoff = pd.Timestamp(CONFIG["cutoff"])
    before = df.InvoiceDate < cutoff
    # Drop every invoice spanning the boundary, so invoice groups never leak.
    crossings = set(df.loc[before, "InvoiceNo"]) & set(df.loc[~before, "InvoiceNo"])
    allowed = ~df.InvoiceNo.isin(crossings)
    return df.loc[before & allowed], df.loc[~before & allowed]


def inject_prices(prices, seed):
    original = np.asarray(prices, dtype=float)
    recorded = original.copy()
    mask = np.zeros(len(original), dtype=bool)
    kinds = np.full(len(original), "clean", dtype=object)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(original), max(1, round(len(original) * CONFIG["corruption_rate"])), replace=False)
    for j, i in enumerate(rows):
        kind = CONFIG["corruptions"][j % 3]
        if kind == "multiply_10":
            recorded[i] *= 10
        elif kind == "divide_10":
            recorded[i] /= 10
        else:
            # Swaps must differ by at least 2x; identical values aren't errors.
            candidates = original[np.abs(np.log(original / original[i])) >= np.log(2)]
            if not len(candidates):
                raise ValueError("Cannot create a meaningful price swap in this sample")
            recorded[i] = rng.choice(candidates)
        mask[i] = True
        kinds[i] = kind
    return recorded, mask, kinds


def sample_data(cache, n_train, seed, ablation):
    df = pd.read_parquet(cache / "transactions.parquet")
    train_pool, test_pool = temporal_split(df)
    if n_train > len(train_pool):
        raise ValueError("Requested training size exceeds the available pool")
    # Prefixes of the same permutation are nested contexts across scale levels.
    rng = np.random.default_rng(seed)
    train = train_pool.iloc[rng.permutation(len(train_pool))[:n_train]].copy()
    test = test_pool.iloc[np.random.default_rng(seed + 10000).permutation(len(test_pool))[:CONFIG["test_rows"]]].copy()
    cols = list(CONFIG["features"])
    if ablation == "no_text":
        cols.remove("Description")
    elif ablation == "no_ids":
        cols = [c for c in cols if c not in ("StockCode", "CustomerID")]
    elif ablation != "full":
        raise ValueError(f"Unknown ablation: {ablation}")
    X_train, X_test = train[cols].copy(), test[cols].copy()
    for c in set(cols) & {"StockCode", "CustomerID", "Country"}:
        # Learn the categories on training data alone; unseen test categories -> NaN.
        dtype = pd.CategoricalDtype(sorted(X_train[c].dropna().unique()))
        X_train[c] = X_train[c].astype(dtype)
        X_test[c] = X_test[c].where(X_test[c].isin(dtype.categories)).astype(dtype)
    original = test.UnitPrice.to_numpy(float)
    recorded, mask, kinds = inject_prices(original, seed)
    digest = hashlib.sha256()
    for arr in [train.index.to_numpy(), test.index.to_numpy(), recorded, mask]:
        digest.update(arr.tobytes())
    meta = {"train_rows": len(train), "test_rows": len(test), "errors": int(mask.sum()),
            "split_sha256": digest.hexdigest(), "training_stock_codes": int(train.StockCode.nunique()),
            "training_customers": int(train.CustomerID.nunique()), "training_descriptions": int(train.Description.nunique()),
            "test_seen_sku_fraction": float(test.StockCode.isin(train.StockCode).mean()),
            "train_last_date": str(train.InvoiceDate.max()), "test_first_date": str(test.InvoiceDate.min()),
            "invoice_groups_disjoint": not bool(set(train.InvoiceNo) & set(test.InvoiceNo)),
            "source_sha256": CONFIG["source_sha256"], "input_features": cols,
            "input_dtypes": {c: str(X_train[c].dtype) for c in cols}}
    return X_train, X_test, train, test, original, recorded, mask, kinds, meta


def metrics(scores, predictions, original, recorded, mask, kinds, low=None, high=None):
    if not np.isfinite(scores).all() or not np.isfinite(predictions).all():
        raise ValueError("Non-finite predictions or anomaly scores")
    k = int(mask.sum())
    selected = np.argsort(-scores, kind="stable")[:k]
    clean = ~mask
    result = {"precision_at_k": float(mask[selected].mean()),
              "auroc": float(roc_auc_score(mask, scores)),
              "clean_log_mae": float(np.abs(np.log(predictions[clean]) - np.log(original[clean])).mean()),
              "clean_price_mae": float(np.abs(predictions[clean] - original[clean]).mean()),
              "injected_error_log_mae_before": float(np.abs(np.log(recorded[mask]) - np.log(original[mask])).mean()),
              "injected_error_log_mae_after": float(np.abs(np.log(predictions[mask]) - np.log(original[mask])).mean()),
              "recall_at_k_by_error": {kind: float(np.isin(np.where(kinds == kind)[0], selected).mean()) for kind in CONFIG["corruptions"]}}
    if low is not None:
        result["clean_80pct_interval_coverage"] = float(((original[clean] >= low[clean]) & (original[clean] <= high[clean])).mean())
    return result


def baseline(train, test, original, recorded, mask, kinds):
    # Strong SKU-specific reference; floor scale at 0.05 log units for stable prices.
    log_price = np.log(train.UnitPrice.to_numpy(float))
    tmp = pd.DataFrame({"sku": train.StockCode.to_numpy(), "log_price": log_price})
    med = tmp.groupby("sku").log_price.median()
    tmp["deviation"] = np.abs(tmp.log_price - tmp.sku.map(med))
    mad = tmp.groupby("sku").deviation.median()
    center = test.StockCode.map(med).fillna(float(np.median(log_price))).to_numpy(float)
    scale = test.StockCode.map(mad).fillna(float(np.median(np.abs(log_price - np.median(log_price))))).to_numpy(float)
    scale = np.maximum(scale * 1.4826, 0.05)
    scores = np.abs(np.log(recorded) - center) / scale
    return metrics(scores, np.exp(center), original, recorded, mask, kinds)


def worker(args):
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(CONFIG["cpu_threads"])
    torch.set_num_interop_threads(1)
    data = sample_data(args.cache, args.rows, args.seed, args.ablation)
    preprocessing = dict(CONFIG["common_preprocessing"])
    if args.profile == "compact_text8":
        preprocessing.update(json.loads(COMPACT_PATH.read_text())["preprocessing_overrides"])
    Xtr, Xte, train, test, original, recorded, mask, kinds, meta = data
    rec = {"status": "running", "model": args.version, "seed": args.seed, "ablation": args.ablation,
           "config_sha256": sha256(CONFIG_PATH), "device": "cpu", "n_estimators_requested": CONFIG["n_estimators"],
           "preprocessing": preprocessing, "runtime_profile": args.profile,
           "runner_sha256": sha256(__file__),
           "runtime_profile_sha256": sha256(COMPACT_PATH) if args.profile == "compact_text8" else None, **meta}
    write_json(args.record, rec)
    try:
        reg = TabPFNRegressor.create_default_for_version(getattr(ModelVersion, args.version), device="cpu",
            n_estimators=CONFIG["n_estimators"], random_state=args.seed,
            n_preprocessing_jobs=1, inference_config=preprocessing)
        rec["checkpoint"] = Path(reg.model_path).name
        rec["checkpoint_sha256"] = sha256(reg.model_path)
        print(f"{args.version}: fit {len(train)} rows ({args.ablation})", flush=True)
        t = time.monotonic()
        reg.fit(Xtr, np.log(train.UnitPrice.to_numpy(float)))
        rec["fit_seconds"] = time.monotonic() - t
        rec["n_estimators_actual"] = reg.n_estimators_
        rec["model_sample_limit"] = reg.inference_config_.MAX_NUMBER_OF_SAMPLES
        rec["model_subsample_samples"] = reg.inference_config_.SUBSAMPLE_SAMPLES
        rec["expanded_feature_count"] = reg.inferred_feature_schema_.num_columns
        write_json(args.record, rec)
        print(f"{args.version}: predict {len(test)} rows; fit {rec['fit_seconds']:.1f}s", flush=True)
        t = time.monotonic()
        o = reg.predict(Xte, output_type="full")
        rec["predict_seconds"] = time.monotonic() - t
        logits = o["logits"]
        y = torch.tensor(np.log(recorded), dtype=logits.dtype, device=logits.device)
        F = o["criterion"].cdf(logits, y.unsqueeze(-1)).squeeze(-1).detach().double().cpu().numpy()
        if not np.isfinite(F).all():
            raise ValueError("Non-finite CDF values")
        scores = -np.log10(np.clip(2 * np.minimum(F, 1 - F), 1e-9, 1))
        predictions = np.exp(np.asarray(o["median"], dtype=float))
        low = np.exp(o["criterion"].icdf(logits, .1).detach().double().cpu().numpy())
        high = np.exp(o["criterion"].icdf(logits, .9).detach().double().cpu().numpy())
        rec["metrics"] = metrics(scores, predictions, original, recorded, mask, kinds, low, high)
        rec["status"] = "ok"
    except Exception as exc:
        rec["status"] = "error"
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)
        print(f"{args.version}: {type(exc).__name__}: {exc}", flush=True)
    write_json(args.record, rec)


def download(versions):
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import _get_model_source, ModelType, get_cache_dir, download_model
    for version in versions:
        v = getattr(ModelVersion, version)
        name = _get_model_source(v, ModelType("regressor")).default_filename
        path = get_cache_dir() / name
        print(f"Preparing cached checkpoint {name}", flush=True)
        status = download_model(path, version=v, which="regressor")
        if status != "ok":
            raise RuntimeError(f"Download failed for {version}: {status}")
        print(f"Ready: {name} ({path.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def run(args):
    import psutil
    if args.profile == "compact_text8" and args.out == Path("results/retail_cpu"):
        args.out = args.out / "compact"

    if not (args.cache / "transactions.parquet").exists():
        raise ValueError("Prepare the dataset first with --prepare")
    args.out.mkdir(parents=True, exist_ok=True)
    hardware = {"platform": platform.platform(), "architecture": platform.machine(),
                "logical_cpu_count": psutil.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False),
                "ram_gib": psutil.virtual_memory().total / 1024**3, "cpu_threads": CONFIG["cpu_threads"]}
    import importlib.metadata as md
    hardware["packages"] = {p: md.version(p) for p in ["tabpfn", "torch", "pandas", "numpy", "skrub", "scikit-learn"]}
    write_json(args.out / "environment.json", hardware)
    for size in args.sizes:
        for seed in args.seeds:
            data = sample_data(args.cache, size, seed, "full")
            _, _, train, test, original, recorded, mask, kinds, meta = data
            bpath = args.out / f"sku_median_{size}_{seed}.json"
            if not bpath.exists():
                t = time.monotonic()
                m = baseline(train, test, original, recorded, mask, kinds)
                write_json(bpath, {"model": "sku_median_mad", "seed": seed, "device": "cpu", "status": "ok",
                                  "config_sha256": sha256(CONFIG_PATH), "runtime_profile": args.profile, **meta, "metrics": m, "seconds": time.monotonic() - t})
            del data, train, test
            for ablation in args.ablations:
                for version in args.versions:
                    record = args.out / f"{version}_{size}_{seed}_{ablation}.json"
                    if record.exists():
                        print(f"Existing record: {record.name}; use a new --out to rerun", flush=True)
                        continue
                    env = os.environ.copy()
                    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
                        env[k] = str(CONFIG["cpu_threads"])
                    cmd = [sys.executable, "-m", "benchmarks.retail_cpu", "--worker", "--version", version,
                           "--rows", str(size), "--seed", str(seed), "--ablation", ablation,
                           "--record", str(record), "--cache", str(args.cache), "--profile", args.profile]
                    log = record.with_suffix(".log")
                    start = time.monotonic()
                    peak = 0
                    stop = None
                    with log.open("w") as f:
                        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                        try:
                            p = psutil.Process(proc.pid)
                            while proc.poll() is None:
                                try:
                                    rss = sum(c.memory_info().rss for c in [p, *p.children(recursive=True)] if c.is_running())
                                    peak = max(peak, rss)
                                except psutil.Error:
                                    pass
                                if time.monotonic() - start > args.timeout:
                                    stop = "timeout"
                                elif peak > CONFIG["rss_limit_gib"] * 1024**3:
                                    stop = "memory_limit"
                                if stop:
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        proc.kill()
                                        proc.wait()
                                    break
                                time.sleep(.5)
                        finally:
                            if proc.poll() is None:
                                proc.kill()
                                proc.wait()
                    rec = json.loads(record.read_text()) if record.exists() else {
                        "model": version, "train_rows": size, "seed": seed, "ablation": ablation,
                        "config_sha256": sha256(CONFIG_PATH), "device": "cpu", "status": "process_error"}
                    if stop:
                        rec["status"] = stop
                    elif rec.get("status") == "running":
                        rec["status"] = "process_error"
                    rec.update({"wall_seconds": time.monotonic() - start, "peak_rss_gib": peak / 1024**3,
                                "exit_code": proc.returncode, "timeout_seconds": args.timeout,
                                "rss_limit_gib": CONFIG["rss_limit_gib"], "recorded_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
                    write_json(record, rec)
                    print(f"{version} {size} {seed} {ablation}: {rec['status']}, {rec['wall_seconds']:.1f}s, {rec['peak_rss_gib']:.2f} GiB", flush=True)
                    if rec.get("metrics"):
                        print(json.dumps(rec["metrics"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("data_cache/online_retail_ii"))
    ap.add_argument("--out", type=Path, default=Path("results/retail_cpu"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--sizes", type=lambda s: list(map(int, s.split(","))), default=[1000])
    ap.add_argument("--seeds", type=lambda s: list(map(int, s.split(","))), default=[1101])
    ap.add_argument("--versions", type=lambda s: s.split(","), default=CONFIG["versions"])
    ap.add_argument("--ablations", type=lambda s: s.split(","), default=["full"])
    ap.add_argument("--timeout", type=int, default=CONFIG["timeout_seconds"])
    ap.add_argument("--profile", choices=["standard", "compact_text8"], default="standard")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", choices=CONFIG["versions"])
    ap.add_argument("--rows", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ablation", choices=CONFIG["ablations"])
    ap.add_argument("--record", type=Path)
    args = ap.parse_args()
    if args.prepare:
        prepare(args.cache, args.out)
    elif args.download:
        download(args.versions)
    elif args.worker:
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
