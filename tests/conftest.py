import pytest


@pytest.fixture
def cpu_inference_threads():
    """Bound CPU threads for the small real-model smoke tests; restore caller settings."""
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(min(4, previous))
    yield
    torch.set_num_threads(previous)
