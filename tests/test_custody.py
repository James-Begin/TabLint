import numpy as np
import pytest

from chainofcustody import Case, Finding, Proof, Suspects
from chainofcustody.cli import main


def make():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    case = Case(X, y, list("abcd"))
    rows = rng.normal(size=(2, 4)).astype(np.float32)
    attack = dict(method="gradient", rows=rows.tolist(), labels=[0, 0], p_before=0.9, p_after=0.1,
                  flipped=True, valid=True, optimization_history=[])
    target = X[:1]
    return case, Finding({"attacks": [attack]}, X, y, target, attack)


def test_report_sections_and_verified_logic():
    case, f = make()
    proof = Proof([{"model": "tabpfn_standard", "p_before": .9, "p_after": .1, "flipped": True, "benign_label_flip": False}])
    assert proof.verified and proof.flipped("tabpfn_standard") and not proof.flipped("hgb")
    s = Suspects([{"row": 20, "delta_p": .5, "signed_delta_p": -.5, "label": 0, "planted": True}], .9, 1, 2, "caveat")
    text = case.report(f, proof, s)
    for part in ("Suspect", "Prove", "Catch", "Limits", "1/2", "not yet verified"):
        assert part in text
    assert "certificate" in text


def test_benign_flip_blocks_verification():
    bad = Proof([{"model": "tabpfn_standard", "flipped": True, "benign_label_flip": True}])
    assert not bad.verified


def test_catch_requires_inputs():
    case, _ = make()
    with pytest.raises(ValueError):
        case.catch()


def test_cli_requires_cuda_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit):
        main(["investigate", "--device", "cuda:0"])
