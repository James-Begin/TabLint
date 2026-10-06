"""Protect the supplied experiment's paired inputs, frozen policies and evidence."""
import json

import numpy as np
import pandas as pd
import pytest

from benchmarks import high_cardinality as experiment
from benchmarks.proofread_confirm import inject
from benchmarks.summarize_high_cardinality import ROOT, load_records, summarize


def test_supplied_results_reproduce_reported_analysis():
    records = load_records(ROOT / "results/high_cardinality", verify_hashes=True)
    result = summarize(records)
    assert result["tables"] == 15 and result["arm_records"] == 60
    h1 = result["metrics"]["precision_at_k"]["h1_version_with_context"]
    assert (h1["wins"], h1["losses"], h1["ties"]) == (9, 2, 4)
    assert h1["mean_difference"] == pytest.approx(.021868707731041614)
    assert h1["wilcoxon_p_one_sided"] == pytest.approx(.01220703125)
    assert result == json.loads((ROOT / "results/high_cardinality/summary.json").read_text())


def test_partial_inventory_and_modified_measurement_rejected(tmp_path):
    files = sorted((ROOT / "results/high_cardinality").glob("*_100*.json"))
    for file in files[:-1]:
        (tmp_path / file.name).write_bytes(file.read_bytes())
    with pytest.raises(ValueError, match="15 paired tables"):
        load_records(tmp_path)
    (tmp_path / files[-1].name).write_bytes(files[-1].read_bytes())
    path = tmp_path / files[0].name
    record = json.loads(path.read_text())
    record["runs"]["v3|hc"]["seconds"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum differs"):
        load_records(tmp_path, verify_hashes=True)


def test_encoding_retains_original_string_policy_and_missing():
    # The measured experiment uses lexical string codes even for numeric IDs.
    encoded = experiment.codes(pd.Series([20, 3, 10, None, 3]))
    np.testing.assert_allclose(encoded, [1, 2, 0, np.nan, 2], equal_nan=True)
    reversed_codes = experiment.codes(pd.Series([3, None, 10, 3, 20]))
    np.testing.assert_allclose(encoded, reversed_codes[::-1], equal_nan=True)


def test_sampling_precedes_exact_shared_injection_and_arms_are_paired():
    spec = experiment.SPECS["employee_salaries"]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(1007, 4)), columns=spec["cont"])
    for col in spec["low"] + spec["high"]:
        frame[col] = [f"category-{i % 71}" for i in range(len(frame))]
    seed = 1002
    expected_rng = np.random.default_rng(seed)
    expected = frame.iloc[expected_rng.choice(len(frame), 1000, replace=False)].reset_index(drop=True)
    expected_corrupt, expected_mask, expected_kind = inject(expected[spec["cont"]].to_numpy(np.float64), .03, expected_rng)
    sample, corrupt, mask, kinds, low, high = experiment.prepare(frame, "employee_salaries", seed)
    pd.testing.assert_frame_equal(sample, expected)
    np.testing.assert_array_equal(corrupt, expected_corrupt)
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(kinds, expected_kind)
    without, low_indices = experiment.context(corrupt, low, high, 1, "num")
    with_context, all_indices = experiment.context(corrupt, low, high, 1, "hc")
    np.testing.assert_array_equal(with_context[:, :without.shape[1]], without)
    np.testing.assert_array_equal(with_context[:, without.shape[1]:], high)
    assert low_indices == [3, 4] and all_indices == [3, 4, 5, 6, 7]
    with pytest.raises(ValueError, match="Unknown context arm"):
        experiment.context(corrupt, low, high, 1, "unknown")


def test_original_model_defaults_are_not_silently_changed(monkeypatch):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    calls = []
    monkeypatch.setattr(TabPFNRegressor, "create_default_for_version", lambda model_version, **kwargs: calls.append((model_version, kwargs)))
    experiment.model("v3.5", 1001, [3, 4, 5])
    assert calls == [(ModelVersion.V3_5, dict(device="cpu", random_state=1001,
        ignore_pretraining_limits=True, categorical_features_indices=[3, 4, 5]))]


def test_auto_mpg_complete_case_filter_and_target_selection():
    frame = experiment.load("auto_mpg", ROOT / "data_cache/unused")
    assert len(frame) == 392
    assert frame[experiment.SPECS["auto_mpg"]["cont"]].notna().all().all()
    assert frame.car.nunique() == 301


def test_new_runner_refuses_product_dependency_drift(monkeypatch, tmp_path):
    import sys
    monkeypatch.setattr(sys, "argv", ["high_cardinality", "--out", str(tmp_path)])
    monkeypatch.setattr(experiment, "version", lambda package: "9.1.0" if package == "tabpfn" else "unused")
    monkeypatch.setattr(experiment, "load", lambda *args: pytest.fail("must reject before a dataset download"))
    with pytest.raises(SystemExit, match="2"):
        experiment.main()


def test_loader_rejects_missing_or_invalid_arm_before_analysis(tmp_path):
    path = ROOT / "results/high_cardinality/auto_mpg_1001.json"
    record = json.loads(path.read_text())
    record["runs"].pop("v3|hc")
    (tmp_path / path.name).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="model/context arms"):
        load_records(tmp_path)
