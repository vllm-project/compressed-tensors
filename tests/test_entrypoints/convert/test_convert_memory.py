# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch
from compressed_tensors.entrypoints.convert.memory import (
    _free_bytes,
    _pick_device,
    exec_jobs_dynamic,
)


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
