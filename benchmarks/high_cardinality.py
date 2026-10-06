"""Portable reproduction of the supplied TabPFN-3.5 high-cardinality CPU study.

This preserves the experiment's TabPFN 9.0.0 defaults, including its category
cap. It does not benchmark the newer product's explicit cap override.
See docs/HIGH_CARDINALITY_RESULTS.md for the distinction and environment.
"""
from __future__ import annotations

import os

# The supplied command set OMP_NUM_THREADS=4; the shared injection module sets
# otherwise-unset BLAS/MKL variables to 2. Preserve that precedence.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from benchmarks.proofread_confirm import inject

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(1001, 1006))
SPECS = {
    "employee_salaries": dict(source=42125, n=1000,
        cont=["current_annual_salary", "2016_gross_pay_received", "2016_overtime_pay", "year_first_hired"],
        low=["gender", "assignment_category"],
        high=["employee_position_title", "division", "department"]),
    "medical_charges": dict(source=42720, n=1000,
        cont=["Total_Discharges", "Average_Covered_Charges", "Average_Total_Payments", "Average_Medicare_Payments"],
        low=[], high=["DRG_Definition", "Provider_State", "Hospital_Referral_Region_(HRR)_Description", "Provider_Id"]),
    "auto_mpg": dict(source=ROOT / "demo/video/auto_mpg.csv", n=None,
        cont=["mpg", "displacement", "horsepower", "weight", "acceleration"],
        low=["cylinders", "model_year", "origin"], high=["car"]),
}


def codes(column):
    """Experiment policy: lexically sorted string identities; missing stays NaN."""
    levels = sorted(column.dropna().astype(str).unique())
    mapping = {value: i for i, value in enumerate(levels)}
    return column.astype(str).where(column.notna()).map(mapping).astype(float).to_numpy()


def load(name, cache):
    spec = SPECS[name]
    if isinstance(spec["source"], int):
        import openml
        openml.config.set_root_cache_directory(str(cache))
        frame, *_ = openml.datasets.get_dataset(spec["source"], download_data=True).get_data(dataset_format="dataframe")
    else:
        frame = pd.read_csv(spec["source"])
    columns = spec["cont"] + spec["low"] + spec["high"]
    return frame[columns].dropna(subset=spec["cont"]).reset_index(drop=True)


def prepare(frame, name, seed):
    spec = SPECS[name]
    rng = np.random.default_rng(seed)
    limit = spec["n"]
    sample = frame if limit is None or len(frame) <= limit else frame.iloc[
        rng.choice(len(frame), limit, replace=False)].reset_index(drop=True)
    corrupt, mask, kinds = inject(sample[spec["cont"]].to_numpy(np.float64), .03, rng)
    n = len(sample)
    low = np.column_stack([codes(sample[c]) for c in spec["low"]]) if spec["low"] else np.zeros((n, 0))
    high = np.column_stack([codes(sample[c]) for c in spec["high"]])
    return sample, corrupt, mask, kinds, low, high


def context(corrupt, low, high, column, arm):
    if arm not in {"num", "hc"}:
        raise ValueError(f"Unknown context arm: {arm}")
    numeric = np.delete(corrupt, column, axis=1)
    matrix = np.c_[numeric, low] if arm == "num" else np.c_[numeric, low, high]
    return matrix, list(range(numeric.shape[1], matrix.shape[1]))


def model(model_name, seed, categorical_indices):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    versions = {"v3": ModelVersion.V3, "v3.5": ModelVersion.V3_5}
    # Deliberately retain the supplied study's defaults; raising the numeric-code
    # category cap would change the measured configuration.
    return TabPFNRegressor.create_default_for_version(
        versions[model_name], device="cpu", random_state=seed,
        ignore_pretraining_limits=True, categorical_features_indices=categorical_indices or None,
    )


def run(frame, name, seed):
    sample, corrupt, mask, kinds, low, high = prepare(frame, name, seed)
    n, d = corrupt.shape
    record = {"dataset": name, "seed": seed, "n": n, "errors": int(mask.sum()),
              "high_card_levels_in_sample": {c: int(sample[c].nunique()) for c in SPECS[name]["high"]},
              "runs": {}}
    for arm in ("num", "hc"):
        for model_name in ("v3", "v3.5"):
            start = time.time()
            scores = np.zeros((n, d))
            for j in range(d):
                matrix, cat = context(corrupt, low, high, j, arm)
                target = corrupt[:, j]
                pit = np.full(n, .5)
                for train, test in KFold(5, shuffle=True, random_state=seed).split(matrix):
                    output = model(model_name, seed, cat).fit(matrix[train], target[train]).predict(matrix[test], output_type="full")
                    logits = output["logits"]
                    observed = torch.tensor(target[test], dtype=logits.dtype, device=logits.device)
                    cdf = output["criterion"].cdf(logits, observed.unsqueeze(-1)).squeeze(-1)
                    pit[test] = np.clip(np.nan_to_num(cdf.detach().double().cpu().numpy().reshape(-1), nan=.5), 1e-9, 1 - 1e-9)
                scores[:, j] = -np.log(2 * np.minimum(pit, 1 - pit))
            flat, truth = scores.ravel(), mask.ravel()
            k = int(truth.sum())
            result = {"precision_at_k": float(truth[np.argsort(-flat, kind="stable")[:k]].mean()),
                      "auroc": float(roc_auc_score(truth, flat)), "seconds": time.time() - start, "by_type": {}}
            for kind in ("decimal", "swap", "offset", "zero", "transposition"):
                positive = kinds.ravel() == kind
                if positive.sum():
                    result["by_type"][kind] = float(roc_auc_score(truth[positive | ~truth], flat[positive | ~truth]))
            record["runs"][f"{model_name}|{arm}"] = result
            print(name, seed, model_name, arm, f"P@k {result['precision_at_k']:.3f} AUROC {result['auroc']:.3f}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(SPECS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--out", type=Path, default=ROOT / "results/reproduction/high_cardinality")
    parser.add_argument("--cache", type=Path, default=ROOT / "data_cache/high_cardinality/openml")
    args = parser.parse_args()
    datasets = args.datasets.split(",")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not datasets or any(name not in SPECS for name in datasets):
        parser.error("Choose datasets from " + ", ".join(SPECS))
    required = {}
    for line in (ROOT / "benchmarks/high_cardinality_protocol/requirements.txt").read_text().splitlines():
        package, expected = line.split("==")
        actual = version(package)
        if actual != expected:
            parser.error(f"{package}=={expected} required for this recorded study; installed {actual}. Use the isolated command in benchmarks/README.md.")
        required[package] = actual
    # Record new reproduction provenance; original uploads lack these sample hashes.
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    environment = {"packages": required, "device": "cpu", "threads": torch.get_num_threads(),
                   "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "datasets": datasets, "seeds": seeds, "sample_hashes": {}}
    env_path = args.out / "environment.json"
    if not env_path.exists() and any(args.out.glob("*.json")):
        parser.error("Output directory has results without reproduction metadata. Choose a fresh --out directory.")
    if env_path.exists():
        prior = json.loads(env_path.read_text())
        if any(prior[key] != environment[key] for key in ("packages", "device", "threads", "runner_sha256", "datasets", "seeds")):
            parser.error("Output directory contains a different configuration. Choose a fresh --out directory.")
        environment = prior
    for name in datasets:
        frame = None
        for seed in seeds:
            destination = args.out / f"{name}_{seed}.json"
            if destination.exists():
                continue
            if frame is None:
                frame = load(name, args.cache)
            sample, *_ = prepare(frame, name, seed)
            environment["sample_hashes"][destination.stem] = hashlib.sha256(sample.to_csv(index=False).encode()).hexdigest()
            env_path.write_text(json.dumps(environment, indent=2) + "\n")
            record = run(frame, name, seed)
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(destination)


if __name__ == "__main__":
    main()
