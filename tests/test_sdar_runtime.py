from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from trajuq.sdar_runtime import (
    CUBLAS_WORKSPACE_CONFIG,
    NORMALIZATION_BACKEND,
    configure_sdar_runtime,
    deterministic_rms_norm,
    validate_sdar_runtime,
)


class Flag:
    def __init__(self, value=False):
        self.value = value


class FakeTorch:
    __version__ = "test"
    version = SimpleNamespace(cuda="test-cuda")

    def __init__(self):
        self._deterministic = False
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )
        self.cuda = SimpleNamespace(is_available=lambda: False)

    def use_deterministic_algorithms(self, value):
        self._deterministic = bool(value)

    def are_deterministic_algorithms_enabled(self):
        return self._deterministic


def test_runtime_fails_closed_on_conflicting_cublas_setting(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_sdar_runtime(FakeTorch())


def test_runtime_validator_rejects_uninstalled_rmsnorm(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    torch = FakeTorch()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    failures = validate_sdar_runtime(
        torch,
        flash_layer_norm=SimpleNamespace(rms_norm_fn=lambda value: value),
    )
    assert failures == ["deterministic Torch RMSNorm is not installed"]


def test_deterministic_rmsnorm_does_not_advance_rng_state():
    torch = pytest.importorskip("torch")
    torch.manual_seed(2027)
    values = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    before = torch.random.get_rng_state().clone()
    output = deterministic_rms_norm(values, weight=torch.ones(3), eps=1e-6)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert output.shape == values.shape


def test_trajectory_observer_does_not_advance_rng_state():
    torch = pytest.importorskip("torch")
    from trajuq.trajectory import TrajectoryAccumulator

    torch.manual_seed(2027)
    logits = torch.randn(1, 2, 7)
    state_ids = torch.tensor([[6, 6]])
    accumulator = TrajectoryAccumulator(prompt_length=0, mask_id=6)
    before = torch.random.get_rng_state().clone()
    accumulator.observe(step=0, logits=logits, state_ids=state_ids)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_runtime_provenance_names_enforced_backend():
    assert NORMALIZATION_BACKEND == "torch_float32_rmsnorm_v1"
    assert os.environ.get("TRAJUQ_ALLOW_NONDETERMINISTIC_SDAR") is None
