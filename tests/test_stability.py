"""CPU tests of the scoped backward patch: restoration, forward identity, finite grads."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tabpfn")

import tabpfn.preprocessing.torch.ops as ops
import tabpfn.preprocessing.torch.torch_standard_scaler as scaler_module
import tabpfn.preprocessing.torch.torch_soft_clip_outliers as outlier_module
from auditkit.stability import stable_backward


def test_bindings_restored_even_on_error():
    s, o = scaler_module.torch_nanstd, outlier_module.torch_nanstd
    with pytest.raises(RuntimeError):
        with stable_backward():
            assert scaler_module.torch_nanstd is not s
            raise RuntimeError("boom")
    assert scaler_module.torch_nanstd is s and outlier_module.torch_nanstd is o


@pytest.mark.parametrize("values", [
    [1.0, 1.0, 1.0, 1.0],                  # zero variance
    [0.0, float("nan"), 2.0, 5.0],          # masked entry
    [0.5, -1.0, 3.0, 2.0],                  # ordinary
])
def test_forward_identical_and_backward_finite(values):
    x = torch.tensor(values, dtype=torch.float32).reshape(4, 1)
    ref = ops.torch_nanstd(x.clone(), axis=0)
    with stable_backward():
        xs = x.clone().requires_grad_(True)
        out = scaler_module.torch_nanstd(xs, axis=0)
        out.sum().backward()
    assert torch.equal(torch.nan_to_num(ref, nan=-7.0), torch.nan_to_num(out.detach(), nan=-7.0))
    finite_inputs = ~torch.isnan(x)
    assert torch.isfinite(xs.grad[finite_inputs]).all()


def test_disabled_is_noop():
    s = scaler_module.torch_nanstd
    with stable_backward(False):
        assert scaler_module.torch_nanstd is s
