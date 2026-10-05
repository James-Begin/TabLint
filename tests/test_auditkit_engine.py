"""CPU-only audit contract tests with a NumPy fake adapter; no model downloads."""
import dataclasses
import json
import subprocess
import sys
import unittest

import numpy as np

from auditkit import AuditConfig, FeatureDomain, audit, candidate_metrics, project


class FakeAdapter:
    """Baseline P1=.8; attack predictions depend on standardized poison values."""
    def __init__(self, n_context, gradient=1.0, nonfinite=False, poison_probability=None):
        self.n_context = n_context
        self.gradient = gradient
        self.nonfinite = nonfinite
        self.poison_probability = poison_probability
        self.calls = []

    def probability(self, X, y, target):
        self.calls.append((X.copy(), y.copy(), target.copy()))
        if len(X) == self.n_context:
            return 0.8
        if self.poison_probability is not None:
            return self.poison_probability
        z = float(np.mean(X[self.n_context:, 0]))
        return float(0.5 + 0.3 * np.tanh(z))

    def value_and_grad(self, X, y, target, n_attack, objective_class):
        gradient = np.zeros_like(X[-n_attack:])
        gradient[:, 0] = np.nan if self.nonfinite else self.gradient
        return self.probability(X, y, target), gradient


class ConfigTests(unittest.TestCase):
    def test_defaults_and_frozen(self):
        cfg = AuditConfig()
        self.assertEqual((cfg.seed, cfg.steps, cfg.lr, cfg.k, cfg.n_estimators, cfg.device),
                         (0, 30, 0.08, 3, 1, "cpu"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.k = 1
        self.assertEqual(AuditConfig(groups=[[0, 1]], protected_indices=[2]).groups, ((0, 1),))

    def test_bad_configs(self):
        bad = [dict(k=0), dict(steps=-1), dict(seed=True), dict(lr=0), dict(lr=np.nan),
               dict(max_real_distance=np.inf), dict(min_target_distance=-1), dict(device="gpu"),
               dict(groups=((0, 1), (1, 2))), dict(groups=((),)), dict(integer_indices=(1, 1)),
               dict(protected_indices=(-1,)), dict(n_estimators=1.5)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AuditConfig(**kwargs)

    def test_pure_import_is_lazy(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; import auditkit; assert 'torch' not in sys.modules; assert 'tabpfn' not in sys.modules"],
            check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class DomainTests(unittest.TestCase):
    def setUp(self):
        # Column 2 is a categorical level absent from context; column 4 is immutable.
        self.x = np.array([[1, 0, 0, 0, 10], [0, 1, 0, 2, 20],
                           [1, 0, 0, 4, 30]], np.float32)
        self.cfg = AuditConfig(groups=((0, 1, 2),), integer_indices=(3,), protected_indices=(4,))
        self.domain = FeatureDomain.fit(self.x, self.cfg)

    def test_context_only_metadata_and_no_mutation(self):
        self.assertTrue(np.array_equal(self.domain.mean, self.x.mean(0, dtype=np.float64)))
        self.assertEqual(self.domain.std[2], 1)
        self.assertEqual(self.domain.upper[2], 0)
        self.assertTrue(np.isfinite(self.domain.nn_radius))
        self.assertFalse(self.domain.context.flags.writeable)
        self.assertTrue(self.x.flags.writeable)
        before = self.domain.metadata()
        candidate_metrics(self.domain, self.x[:1], np.full(5, 1000), self.x[:1])
        self.assertEqual(before, self.domain.metadata())

    def test_projection_and_validation(self):
        seed = self.x[:1]
        raw = np.array([[0.2, 0.9, 99, 2.7, 99]])
        snapped = project(self.domain, raw, seed)
        np.testing.assert_array_equal(snapped, [[0, 1, 0, 3, 10]])
        metrics = candidate_metrics(self.domain, snapped, self.x[1], seed)
        self.assertTrue(metrics["valid"])
        self.assertTrue(metrics["protected_unchanged"])
        self.assertTrue(metrics["integer_valid"])
        invalid = snapped.copy()
        invalid[0, :3] = [0, 0, 1]
        self.assertTrue(candidate_metrics(self.domain, invalid, self.x[1], seed)["category_valid"])
        self.assertFalse(candidate_metrics(self.domain, invalid, self.x[1], seed)["in_bounds"])
        invalid[0, 3:] = [1.5, 20]
        m = candidate_metrics(self.domain, invalid, self.x[1], seed)
        self.assertFalse(m["integer_valid"])
        self.assertFalse(m["protected_unchanged"])
        self.assertFalse(m["valid"])

    def test_partial_protected_category(self):
        domain = FeatureDomain.fit(self.x, AuditConfig(groups=((0, 1, 2),), protected_indices=(0,)))
        raw = np.array([[0, 1, 10, 3, 10], [1, 0, 10, 3, 20]], np.float32)
        out = domain.project(raw, self.x[:2])
        np.testing.assert_array_equal(out[:, :3], self.x[:2, :3])
        self.assertTrue(domain.validate(out, self.x[2], self.x[:2])["valid"])

    def test_empty_is_invalid(self):
        m = self.domain.validate(np.empty((0, 5)), self.x[0], np.empty((0, 5)))
        self.assertFalse(m["valid"])
        self.assertIsNone(m["dtarget_min"])
        self.assertIsNone(m["dnn_max"])
        json.dumps(m, allow_nan=False)

    def test_explicit_integer_not_inferred(self):
        plain = FeatureDomain.fit([[0], [2]])
        self.assertTrue(plain.validate([[0.5]], [1], [[0]])["valid"])
        declared = FeatureDomain.fit([[0], [2]], AuditConfig(integer_indices=(0,)))
        self.assertFalse(declared.validate([[0.5]], [1], [[0]])["valid"])
        with self.assertRaises(ValueError):
            FeatureDomain.fit([[0.5], [2]], AuditConfig(integer_indices=(0,)))

    def test_nearest_distance_and_radius(self):
        d = FeatureDomain.fit([[0], [2], [4]])
        self.assertAlmostEqual(d.nn_radius, 2 / np.std([0, 2, 4]))
        m = d.validate([[1]], [0], [[0]])
        self.assertAlmostEqual(m["dnn_max"], 1 / np.std([0, 2, 4]))
        self.assertAlmostEqual(m["dtarget_min"], 1 / np.std([0, 2, 4]))
        duplicate = FeatureDomain.fit([[1], [1]])
        self.assertEqual(duplicate.nn_radius, 0)

    def test_bad_domain_and_arrays(self):
        for x, config in [([[1, 1], [0, 1]], AuditConfig(groups=((0, 1),))),
                          ([[0], [1]], AuditConfig(protected_indices=(1,))),
                          ([[np.inf], [0]], AuditConfig()), ([[0]], AuditConfig()),
                          ([["0"], ["1"]], AuditConfig())]:
            with self.subTest(x=x), self.assertRaises(ValueError):
                FeatureDomain.fit(x, config)
        with self.assertRaises(ValueError):
            FeatureDomain.fit(self.x, feature_names=["wrong"])


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0, 10], [2, 20], [4, 30], [6, 40]], np.float32)
        self.y = np.array([0, 0, 1, 1])

    def run_audit(self, **kwargs):
        cfg = kwargs.pop("config", AuditConfig(k=2, steps=3, lr=0.1))
        adapter = kwargs.pop("adapter", FakeAdapter(len(self.x)))
        return audit(self.x, self.y, [1, 15], config=cfg, adapter=adapter, **kwargs)

    def test_schema_and_standardized_graph(self):
        fake = FakeAdapter(len(self.x))
        report = self.run_audit(adapter=fake, feature_names=["a", "b"])
        json.dumps(report, allow_nan=False)
        self.assertNotIn("dataset", report)
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(set(report), {"schema_version", "config", "baseline", "attacks", "warnings", "diagnostics"})
        self.assertTrue(report["baseline"]["repeatable"])
        self.assertEqual(report["diagnostics"]["grad_model"], "tabpfn_3_5_differentiable_surrogate")
        np.testing.assert_allclose(fake.calls[0][0], (self.x - self.x.mean(0)) / self.x.std(0), rtol=1e-6)
        np.testing.assert_allclose(fake.calls[0][2], (np.array([[1, 15]]) - self.x.mean(0)) / self.x.std(0), rtol=1e-6)
        required = {"method", "n_rows", "rows", "labels", "p_before", "p_after", "flipped", "valid",
                    "constraints", "optimization_history", "evaluations", "elapsed_s", "masked_gradient_entries"}
        self.assertEqual([a["method"] for a in report["attacks"]],
                         ["exact_duplicates", "random_real", "near_rows", "gradient"])
        for attack in report["attacks"]:
            self.assertTrue(required <= set(attack))
            self.assertEqual(attack["n_rows"], len(attack["rows"]))
            self.assertIsInstance(attack["evaluations"], int)
        self.assertTrue(report["attacks"][0]["unconstrained"])
        real = report["attacks"][1]
        np.testing.assert_array_equal(real["labels"], 1 - self.y[real["source_indices"]])
        near, grad = report["attacks"][2:]
        self.assertEqual(grad["optimization_history"][0]["p_after"], near["p_after"])
        self.assertEqual(grad["optimization_history"][0]["step"], 0)
        # Fake gradient increases P1, the WRONG direction for this baseline.
        self.assertEqual(grad["rows"], near["rows"])
        self.assertEqual(grad["p_after"], near["p_after"])
        self.assertFalse(grad["aborted"])

    def test_nonfinite_aborts_preserving_init(self):
        report = self.run_audit(adapter=FakeAdapter(len(self.x), nonfinite=True))
        near, grad = report["attacks"][2:]
        self.assertEqual(grad["rows"], near["rows"])
        self.assertEqual(grad["completed_steps"], 0)
        self.assertEqual(grad["nonfinite_gradient_entries"], 2)
        self.assertEqual(grad["masked_gradient_entries"], 0)
        self.assertTrue(grad["aborted"])
        self.assertTrue(any("none were masked" in w for w in report["warnings"]))
        json.dumps(report, allow_nan=False)

    def test_invalid_flip_is_not_success_or_certificate(self):
        report = self.run_audit(config=AuditConfig(k=2, steps=2, min_target_distance=1000),
                                adapter=FakeAdapter(len(self.x), poison_probability=0.1))
        grad = report["attacks"][-1]
        self.assertTrue(grad["prediction_changed"])
        self.assertFalse(grad["valid"])
        self.assertFalse(grad["flipped"])
        self.assertEqual(grad["valid_evaluations"], 0)
        self.assertFalse(report["diagnostics"]["robustness_certificate"])
        self.assertTrue(any("zero valid" in w for w in report["warnings"]))

    def test_zero_steps_and_protected_features(self):
        report = self.run_audit(config=AuditConfig(k=2, steps=0, protected_indices=(0, 1)))
        near, grad = report["attacks"][2:]
        self.assertEqual(near["rows"], grad["rows"])
        self.assertEqual(grad["evaluations"], 0)
        self.assertEqual(grad["optimization_history"][0]["step"], 0)
        self.assertTrue(grad["constraints"]["protected_unchanged"])
        np.testing.assert_array_equal(near["rows"], self.x[near["seed_indices"]])

    def test_invalid_step_cannot_replace_valid_initialization(self):
        # Moving off an exact context row violates a zero NN-distance cap.
        report = self.run_audit(config=AuditConfig(k=1, steps=1, max_real_distance=0),
                                adapter=FakeAdapter(len(self.x), gradient=1))
        near, grad = report["attacks"][2:]
        self.assertTrue(near["valid"])
        self.assertEqual(grad["rows"], near["rows"])
        self.assertFalse(grad["optimization_history"][1]["valid"])

    def test_valid_step_replaces_invalid_initialization(self):
        report = audit([[0], [2]], [0, 1], [0],
                       config=AuditConfig(k=1, steps=1, lr=2, min_target_distance=1.9),
                       adapter=FakeAdapter(2, gradient=10))
        near, grad = report["attacks"][2:]
        self.assertFalse(near["valid"])
        self.assertTrue(grad["valid"])
        self.assertEqual(grad["rows"], [[2.0]])

    def test_integer_small_updates_accumulate(self):
        report = audit([[0], [4]], [0, 1], [4],
                       config=AuditConfig(k=1, steps=4, lr=0.1, integer_indices=(0,)),
                       adapter=FakeAdapter(2, gradient=-1))
        near, grad = report["attacks"][2:]
        self.assertEqual(near["rows"], [[4.0]])
        self.assertEqual(grad["rows"], [[3.0]])
        self.assertLess(grad["p_after"], near["p_after"])
        self.assertTrue(grad["constraints"]["integer_valid"])

    def test_repeatability_and_gradient_exception_are_disclosed(self):
        class Unstable(FakeAdapter):
            def probability(self, X, y, target):
                value = super().probability(X, y, target)
                if len(self.calls) == 2:
                    return 0.7
                return value

            def value_and_grad(self, *args):
                raise RuntimeError("gradient unavailable")
        report = self.run_audit(adapter=Unstable(len(self.x)))
        self.assertFalse(report["baseline"]["repeatable"])
        self.assertTrue(report["attacks"][-1]["aborted"])
        self.assertTrue(any("gradient unavailable" in w for w in report["warnings"]))
        self.assertEqual(report["attacks"][-1]["rows"], report["attacks"][-2]["rows"])

    def test_input_validation(self):
        fake = FakeAdapter(len(self.x))
        for y, target in [([0, 0, 0, 0], [1, 2]), ([0, 1, 2, 1], [1, 2]),
                          ([0, 0, 1.00000001, 1], [1, 2]),
                          ([[0], [0], [1], [1]], [1, 2]), (self.y, [[1], [2]]),
                          (self.y, [np.nan, 2])]:
            with self.subTest(y=y, target=target), self.assertRaises(ValueError):
                audit(self.x, y, target, adapter=fake)
        self.assertFalse(fake.calls)
        with self.assertRaises(ValueError):
            audit(self.x, self.y, [1, 2], config={}, adapter=fake)
        report = audit(self.x, self.y.astype(bool), [[1, 2]],
                       config=AuditConfig(steps=0), adapter=fake)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()


def test_stable_backward_flag_validated_and_recorded():
    import pytest
    with pytest.raises(ValueError):
        AuditConfig(stable_backward="yes")
    assert AuditConfig(stable_backward=True).stable_backward is True
