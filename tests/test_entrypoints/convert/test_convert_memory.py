# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import inspect
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.entrypoints.convert.memory import (
    _FALLBACK_MULTIPLIER,
    TensorProfiler,
    _free_bytes,
    _pick_device,
    estimate_job_memory,
    exec_jobs_dynamic,
)
from tests.testing_utils import requires_gpu


_LOAD_TARGET = (
    "compressed_tensors.entrypoints.convert.memory."
    "load_tensors_from_inverse_weight_map"
)


# ── estimate_job_memory ────────────────────────────────────────────────


class _MetaConverter:
    def validate(self, tensors):
        weight = tensors["weight"]
        return {"weight_packed": torch.empty(weight.shape, dtype=torch.int8)}


class _RaisingConverter:
    def validate(self, tensors):
        raise ValueError("incompatible")


def test_estimate_job_memory_profiles_on_meta():
    inverse_weight_map = {"/source/model.safetensors": ["weight"]}

    def fake_load(*args, **kwargs):
        return {"weight": torch.empty(1024, dtype=torch.float32)}

    with torch.device("meta"):
        with patch(_LOAD_TARGET, side_effect=fake_load) as load_tensors:
            estimate = estimate_job_memory(inverse_weight_map, [_MetaConverter()])

    load_tensors.assert_called_once_with(inverse_weight_map, device="meta")
    assert estimate == 1024 * 4 + 1024 * 1


def test_estimate_job_memory_falls_back_on_profiler_failure():
    inverse_weight_map = {"/source/model.safetensors": ["weight"]}

    def fake_load(*args, **kwargs):
        return {"weight": torch.empty(1024, dtype=torch.float32)}

    with torch.device("meta"):
        with patch(_LOAD_TARGET, side_effect=fake_load):
            estimate = estimate_job_memory(inverse_weight_map, [_RaisingConverter()])

    assert estimate == int(1024 * 4 * _FALLBACK_MULTIPLIER)


# ── _free_bytes ────────────────────────────────────────────────────────


def test_free_bytes_subtracts_reserved():
    dev = torch.device("cuda:0")
    assert (
        _free_bytes(dev, {dev: 10_000_000_000}, {dev: 3_000_000_000}) == 7_000_000_000
    )


def test_free_bytes_clamps_to_zero():
    dev = torch.device("cuda:0")
    assert _free_bytes(dev, {dev: 1000}, {dev: 5000}) == 0


def test_free_bytes_unknown_device_returns_zero():
    assert _free_bytes(torch.device("cpu"), {}, {}) == 0


# ── _pick_device ───────────────────────────────────────────────────────


def test_pick_most_free_device():
    d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
    assert _pick_device([d0, d1], 1000, {d0: 40e9, d1: 80e9}, {d0: 0, d1: 0}) == d1


def test_pick_none_when_nothing_fits():
    d0 = torch.device("cuda:0")
    assert _pick_device([d0], 2000, {d0: 1000}, {d0: 0}) is None


def test_pick_respects_reservations():
    d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
    # d0: 80 GB - 70 GB reserved = 10 GB available; d1: 50 GB - 0 = 50 GB
    # job needs 20 GB -> pick d1
    assert _pick_device([d0, d1], 20e9, {d0: 80e9, d1: 50e9}, {d0: 70e9, d1: 0}) == d1


def test_pick_skips_cpu_devices():
    cpu = torch.device("cpu")
    assert _pick_device([cpu], 1000, {}, {cpu: 0}) is None


# ── exec_jobs_dynamic: CPU path (no GPU required) ──────────────────────


def test_cpu_path_runs_all_jobs():
    results = exec_jobs_dynamic(
        jobs=[lambda dev: dev for _ in range(5)],
        devices=[torch.device("cpu")],
        max_workers=2,
        memory_estimates=[1000] * 5,
    )
    assert len(results) == 5
    assert all(r == torch.device("cpu") for r in results)


def test_cpu_path_empty_jobs():
    assert exec_jobs_dynamic([], [torch.device("cpu")], 1, []) == []


def test_cpu_path_preserves_order():
    jobs = [lambda dev, i=i: i for i in range(10)]
    out = exec_jobs_dynamic(jobs, [torch.device("cpu")], 4, [100] * 10)
    assert out == list(range(10))


# ── exec_jobs_dynamic: input validation ───────────────────────────────


def test_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="memory_estimates length"):
        exec_jobs_dynamic([lambda dev: None], [torch.device("cpu")], 1, [100, 200])


def test_raises_on_negative_estimate():
    with pytest.raises(ValueError, match="negative"):
        exec_jobs_dynamic([lambda dev: None], [torch.device("cpu")], 1, [-1])


def test_raises_on_max_workers_below_one():
    with pytest.raises(ValueError, match="max_workers"):
        exec_jobs_dynamic([lambda dev: None], [torch.device("cpu")], 0, [100])


def test_raises_on_empty_devices_with_jobs():
    with pytest.raises(ValueError, match="devices must not be empty"):
        exec_jobs_dynamic([lambda dev: None], [], 1, [100])


# ── exec_jobs_dynamic: error handling ─────────────────────────────────


_PATCH_TARGET = (
    "compressed_tensors.entrypoints.convert.memory"
    ".torch.accelerator.memory.get_memory_info"
)


@patch(_PATCH_TARGET)
def test_raises_when_no_device_fits(mock_mem_info):
    mock_mem_info.return_value = (1000, 96_000_000_000)
    with pytest.raises(RuntimeError, match="No device has enough"):
        exec_jobs_dynamic(
            jobs=[lambda dev: None],
            devices=[torch.device("cuda:0")],
            max_workers=2,
            memory_estimates=[10_000_000_000],
        )


@patch(_PATCH_TARGET)
def test_single_worker_raises_when_job_exceeds_capacity(mock_mem_info):
    mock_mem_info.return_value = (1000, 96_000_000_000)
    with pytest.raises(RuntimeError, match="exceeds estimated capacity"):
        exec_jobs_dynamic(
            jobs=[lambda dev: None],
            devices=[torch.device("cuda:0")],
            max_workers=1,
            memory_estimates=[10_000_000_000],
        )


# ── TensorProfiler ─────────────────────────────────────────────────────


def _n_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.nbytes for tensor in tensors)


def device_parametrize(test):
    signature = inspect.signature(test)

    @functools.wraps(test)
    def wrapper(*args, _device, **kwargs):
        if _device == "cuda" and not torch.accelerator.is_available():
            pytest.skip("CUDA unavailable")

        with torch.device(_device):
            return test(*args, **kwargs)

    wrapper.__signature__ = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter("_device", inspect.Parameter.KEYWORD_ONLY),
        ]
    )

    return pytest.mark.parametrize("_device", ["meta", "cpu", "cuda"])(wrapper)


@device_parametrize
def test_profiler_constructor():
    with TensorProfiler() as prof:
        a = torch.Tensor([0 for _ in range(16)])

    assert prof.memory["total"] == _n_bytes(a)


@device_parametrize
def test_profiler_constructor_functions():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = torch.zeros(16)
        c = torch.ones(16)
        d = torch.full((16,), 0)

    assert prof.memory["total"] == _n_bytes(a, b, c, d)


@device_parametrize
def test_profiler_operations():
    with TensorProfiler() as prof:
        a = torch.Tensor([1 for _ in range(16)])
        b = a + a

    assert prof.memory["total"] == _n_bytes(a, b)


@device_parametrize
def test_profiler_views():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        a_storage_bytes = _n_bytes(a)

        b = a[:8]
        c = a[8:]
        d = a[4:12]  # noqa: F841

        del a
        assert prof.memory["total"] == a_storage_bytes

        del b, c
        assert prof.memory["total"] == a_storage_bytes


@requires_gpu
def test_profiler_device_movement():
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")
    meta_device = torch.device("meta")

    with TensorProfiler() as prof:
        a = torch.empty(16, device=cpu_device)
        b = a.to(device=gpu_device)
        c = a.to(device=meta_device)

    assert prof.memory["total"] == _n_bytes(a, b, c)
    assert prof.memory[cpu_device] == _n_bytes(a)
    assert prof.memory[gpu_device] == _n_bytes(b)
    assert prof.memory[meta_device] == _n_bytes(c)


@device_parametrize
def test_profiler_dtype_movement():
    with TensorProfiler() as prof:
        a = torch.empty(16, dtype=torch.float32)
        b = a.to(dtype=torch.bfloat16)
        c = a.to(dtype=torch.float8_e4m3fn)

    assert prof.memory["total"] == _n_bytes(a, b, c)


@device_parametrize
def test_profiler_complex_operations():
    with TensorProfiler() as prof:
        a = torch.randn(32)
        b = torch.randn(32)
        c = a * b
        d = torch.sin(c)
        e = torch.cat([a, b, c, d])

    assert prof.memory["total"] == _n_bytes(a, b, c, d, e)


@device_parametrize
def test_profiler_deletion_tracking():
    with TensorProfiler() as prof:
        a = torch.randn(64)
        b = torch.randn(64)
        c = a + b
        del a
        d = c * 2
        del c
        e = torch.zeros_like(b)
        del b

    assert prof.memory["total"] == _n_bytes(d, e)


@device_parametrize
def test_profiler_view_operations():
    with TensorProfiler() as prof:
        a = torch.randn(4, 4)
        b = a.view(16)  # noqa: F841
        c = a.reshape(2, 8)  # noqa: F841
        d = a.t()  # noqa: F841

    assert prof.memory["total"] == _n_bytes(a)


@device_parametrize
def test_profiler_different_dtypes():
    with TensorProfiler() as prof:
        a = torch.ones(16, dtype=torch.float32)
        b = torch.ones(16, dtype=torch.float64)
        c = torch.ones(16, dtype=torch.int32)
        d = torch.ones(16, dtype=torch.bool)

    assert prof.memory["total"] == _n_bytes(a, b, c, d)


@device_parametrize
def test_profiler_inplace_operations():
    with TensorProfiler() as prof:
        a = torch.randn(16)
        b = torch.randn(16)
        a.add_(b)
        b.mul_(2)

    assert prof.memory["total"] == _n_bytes(a, b)


def test_profiler_catches_exception():
    with TensorProfiler() as prof:
        raise ValueError("boom")

    assert isinstance(prof.exception, ValueError)
