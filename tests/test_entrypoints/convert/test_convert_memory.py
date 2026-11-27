# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import inspect
import threading
from concurrent.futures import wait
from contextlib import contextmanager
from threading import get_ident, local
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.entrypoints.convert.memory import (
    _FALLBACK_MULTIPLIER,
    TensorProfiler,
    _cgroup_available_bytes,
    _free_bytes,
    _host_available_bytes,
    _pick_device,
    _run_job_on_device,
    _snapshot_free,
    estimate_job_memory,
    exec_jobs_dynamic,
)
from tests.testing_utils import requires_gpu


_LOAD_TARGET = (
    "compressed_tensors.entrypoints.convert.memory."
    "load_tensors_from_inverse_weight_map"
)
_HOST_TARGET = "compressed_tensors.entrypoints.convert.memory._host_available_bytes"
_WAIT_TARGET = "compressed_tensors.entrypoints.convert.memory.wait"
_PATCH_TARGET = (
    "compressed_tensors.entrypoints.convert.memory"
    ".torch.accelerator.memory.get_memory_info"
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


# ── worker device context ─────────────────────────────────────────────


def test_run_job_selects_assigned_accelerator_device():
    events = []

    class DeviceContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")

    device = torch.device("cuda:3")

    def job(dev):
        assert events == ["enter"]
        assert dev == device
        return "passed"

    with patch(
        "compressed_tensors.entrypoints.convert.memory.torch.accelerator.device_index",
        return_value=DeviceContext(),
    ) as device_index:
        assert _run_job_on_device(job, device) == "passed"

    device_index.assert_called_once_with(3)
    assert events == ["enter", "exit"]


def test_run_job_does_not_enter_device_context_for_cpu():
    device = torch.device("cpu")

    with patch(
        "compressed_tensors.entrypoints.convert.memory.torch.accelerator.device_index"
    ) as device_index:
        assert _run_job_on_device(lambda dev: dev, device) == device

    device_index.assert_not_called()


def test_dynamic_scheduler_selects_device_in_worker_threads():
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    worker_state = local()
    main_thread = get_ident()

    @contextmanager
    def select_device(index):
        assert not hasattr(worker_state, "device_index")
        worker_state.device_index = index
        try:
            yield
        finally:
            del worker_state.device_index

    def make_job(job_id):
        def job(device):
            assert get_ident() != main_thread
            assert worker_state.device_index == device.index
            return job_id, device

        return job

    with (
        patch(_PATCH_TARGET, return_value=(100, 100)),
        patch(
            "compressed_tensors.entrypoints.convert.memory."
            "torch.accelerator.device_index",
            side_effect=select_device,
        ) as device_index,
    ):
        results = exec_jobs_dynamic(
            jobs=[make_job(0), make_job(1)],
            devices=devices,
            max_workers=2,
            memory_estimates=[60, 60],
        )

    assert results == [(0, devices[0]), (1, devices[1])]
    assert {call.args[0] for call in device_index.call_args_list} == {0, 1}


# ── exec_jobs_dynamic: CPU path (no GPU required) ──────────────────────


@patch(_HOST_TARGET, return_value=10**12)
def test_cpu_path_runs_all_jobs(_):
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
    assert exec_jobs_dynamic([], [], 1, []) == []


@patch(_HOST_TARGET, return_value=10**12)
def test_cpu_path_preserves_order(_):
    jobs = [lambda dev, i=i: i for i in range(10)]
    out = exec_jobs_dynamic(jobs, [torch.device("cpu")], 4, [100] * 10)
    assert out == list(range(10))


def _max_admitted(devices, max_workers, estimates):
    """Run cpu jobs and return the most jobs the scheduler had admitted at once"""
    admitted = []

    def recording_wait(futures, **kwargs):
        admitted.append(len(futures))
        return wait(futures, **kwargs)

    with patch(_WAIT_TARGET, side_effect=recording_wait):
        out = exec_jobs_dynamic(
            [lambda dev: dev] * len(estimates), devices, max_workers, estimates
        )
    assert out == [torch.device("cpu")] * len(estimates)
    return max(admitted)


@patch(_HOST_TARGET, return_value=10**12)
def test_cpu_admits_up_to_max_workers_jobs(_):
    assert _max_admitted([torch.device("cpu")], 2, [1] * 6) == 2


@patch(_HOST_TARGET, return_value=1000)
def test_cpu_admission_is_limited_by_host_memory(_):
    # only three 300-byte jobs fit in 1000 bytes, despite max_workers=4
    assert _max_admitted([torch.device("cpu")], 4, [300] * 6) == 3


@patch(_HOST_TARGET, return_value=1000)
def test_cpu_aliases_share_one_host_memory_budget(_):
    devices = [torch.device("cpu"), torch.device("cpu:0")]
    assert _max_admitted(devices, 2, [800, 800]) == 1


@patch(_HOST_TARGET, return_value=10**12)
def test_cpu_jobs_run_concurrently(_):
    # each job waits for the other, so both must run at the same time to finish
    barrier = threading.Barrier(2, timeout=5)
    jobs = [lambda dev: barrier.wait()] * 2
    assert sorted(exec_jobs_dynamic(jobs, [torch.device("cpu")], 2, [1, 1])) == [0, 1]


@pytest.mark.parametrize("max_workers", (1, 2))
@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@patch(_HOST_TARGET, return_value=100)
@patch(_PATCH_TARGET, return_value=(100, 1000))
def test_raises_before_running_any_job_if_a_job_cannot_fit(
    _memory_info, _host, device, max_workers
):
    ran = []
    jobs = [lambda dev: ran.append(0), lambda dev: ran.append(1)]
    with pytest.raises(RuntimeError, match="Job 1 needs an estimated"):
        exec_jobs_dynamic(jobs, [torch.device(device)], max_workers, [10, 1000])
    assert ran == []


# ── host memory ────────────────────────────────────────────────────────


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _proc_files(tmp_path, cgroup, *mounts):
    """
    Write /proc/self/cgroup and /proc/self/mountinfo stand-ins. Each mount is
    (fs-type, root, mount point, super options).
    """
    _write(tmp_path / "proc" / "cgroup", cgroup)
    _write(
        tmp_path / "proc" / "mountinfo",
        "".join(
            f"{30 + i} 1 0:{30 + i} {root} {point} rw shared:{i} - {fs} {fs} {opts}\n"
            for i, (fs, root, point, opts) in enumerate(mounts)
        ),
    )
    return tmp_path / "proc" / "cgroup", tmp_path / "proc" / "mountinfo"


def test_cgroup_v2_available_bytes(tmp_path):
    cg = tmp_path / "cg"
    _write(cg / "memory.max", "1000\n")
    _write(cg / "memory.current", "600\n")
    _write(cg / "memory.stat", "anon 400\ninactive_file 100\n")
    files = _proc_files(tmp_path, "0::/\n", ("cgroup2", "/", cg, "rw"))
    # limit - (usage - reclaimable inactive file cache)
    assert _cgroup_available_bytes(*files) == 500


def test_cgroup_v2_takes_smallest_headroom_over_ancestors(tmp_path):
    cg = tmp_path / "cg"
    _write(cg / "a" / "b" / "memory.max", "max\n")
    _write(cg / "a" / "b" / "memory.current", "300\n")
    _write(cg / "a" / "memory.max", "800\n")
    _write(cg / "a" / "memory.current", "700\n")
    files = _proc_files(tmp_path, "0::/a/b\n", ("cgroup2", "/", cg, "rw"))
    assert _cgroup_available_bytes(*files) == 100


def test_cgroup_v2_mount_of_a_subtree(tmp_path):
    # the mount exposes /tenant, so the process's /tenant/job is cg/job, whose
    # tighter limit must not be missed
    cg = tmp_path / "cg"
    _write(cg / "memory.max", "8000\n")
    _write(cg / "memory.current", "1000\n")
    _write(cg / "job" / "memory.max", "1500\n")
    _write(cg / "job" / "memory.current", "500\n")
    files = _proc_files(tmp_path, "0::/tenant/job\n", ("cgroup2", "/tenant", cg, "rw"))
    assert _cgroup_available_bytes(*files) == 1000


@pytest.mark.parametrize(
    ("cgroup", "root"),
    (
        ("/other", "/tenant"),
        # mount roots and paths with ".." are outside this cgroup namespace
        ("/", "/.."),
        ("/../other", "/"),
        ("/..", "/.."),
    ),
)
def test_cgroup_not_visible_through_mount(tmp_path, cgroup, root):
    cg = tmp_path / "cg"
    _write(cg / "memory.max", "1000\n")
    _write(cg / "memory.current", "600\n")
    files = _proc_files(tmp_path, f"0::{cgroup}\n", ("cgroup2", root, cg, "rw"))
    assert _cgroup_available_bytes(*files) is None


@pytest.mark.parametrize("first_view_limit", (None, 8000))
def test_cgroup_takes_smallest_headroom_over_mount_views(tmp_path, first_view_limit):
    # two mounts of the same hierarchy: the first shows the process's cgroup but
    # not its memory files (and maybe a looser parent limit), the second shows them
    first, second = tmp_path / "first", tmp_path / "second"
    (first / "job").mkdir(parents=True)
    if first_view_limit is not None:
        _write(first / "memory.max", f"{first_view_limit}\n")
        _write(first / "memory.current", "1000\n")
    _write(second / "job" / "memory.max", "1500\n")
    _write(second / "job" / "memory.current", "500\n")
    files = _proc_files(
        tmp_path,
        "0::/job\n",
        ("cgroup2", "/", first, "rw"),
        ("cgroup2", "/", second, "rw"),
    )
    assert _cgroup_available_bytes(*files) == 1000


def test_cgroup_mountinfo_escapes(tmp_path):
    # mountinfo writes spaces in the mount root and mount point as "\040"
    cg = tmp_path / "cgroup mount"
    _write(cg / "task" / "memory.max", "1500\n")
    _write(cg / "task" / "memory.current", "500\n")
    point = str(cg).replace(" ", "\\040")
    files = _proc_files(
        tmp_path, "0::/my job/task\n", ("cgroup2", "/my\\040job", point, "rw")
    )
    assert _cgroup_available_bytes(*files) == 1000


def test_cgroup_v2_without_limit(tmp_path):
    cg = tmp_path / "cg"
    _write(cg / "memory.max", "max\n")
    _write(cg / "memory.current", "600\n")
    files = _proc_files(tmp_path, "0::/\n", ("cgroup2", "/", cg, "rw"))
    assert _cgroup_available_bytes(*files) is None


@pytest.mark.parametrize("controllers", ("memory", "cpu,memory"))
def test_cgroup_v1_memory_controller(tmp_path, controllers):
    # hybrid hierarchy: the v1 memory controller, possibly mounted together with
    # other controllers, takes precedence over the v2 hierarchy
    v1, v2 = tmp_path / controllers, tmp_path / "unified"
    leaf = v1 / "docker" / "abc"
    _write(leaf / "memory.limit_in_bytes", "2000\n")
    _write(leaf / "memory.usage_in_bytes", "1500\n")
    _write(leaf / "memory.stat", "total_inactive_file 500\n")
    _write(v2 / "memory.max", "100\n")
    _write(v2 / "memory.current", "0\n")
    files = _proc_files(
        tmp_path,
        f"12:{controllers}:/docker/abc\n0::/\n",
        ("cgroup2", "/", v2, "rw"),
        ("cgroup", "/", v1, f"rw,{controllers}"),
    )
    assert _cgroup_available_bytes(*files) == 1000


def test_cgroup_unavailable(tmp_path):
    assert _cgroup_available_bytes(tmp_path / "cgroup", tmp_path / "mountinfo") is None


@pytest.mark.parametrize(("cgroup", "expected"), ((32, 32), (None, 512)))
def test_host_available_bytes_capped_by_cgroup(cgroup, expected):
    module = "compressed_tensors.entrypoints.convert.memory"
    with (
        patch(f"{module}.psutil.virtual_memory") as virtual_memory,
        patch(f"{module}._cgroup_available_bytes", return_value=cgroup),
    ):
        virtual_memory.return_value.available = 512
        assert _host_available_bytes() == expected


@patch(_HOST_TARGET, return_value=123)
def test_snapshot_free_cpu_only_reports_host_memory(_):
    cpu = torch.device("cpu")
    assert _snapshot_free([cpu]) == {cpu: 123}


@patch(_HOST_TARGET, return_value=123)
@patch(_PATCH_TARGET, return_value=(456, 1000))
def test_snapshot_free_skips_cpu_alongside_accelerators(*_):
    cpu, cuda = torch.device("cpu"), torch.device("cuda:0")
    assert _snapshot_free([cuda, cpu]) == {cuda: 456}


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


@patch(_PATCH_TARGET)
def test_raises_when_no_device_fits(mock_mem_info):
    mock_mem_info.return_value = (1000, 96_000_000_000)
    with pytest.raises(RuntimeError, match="exceeds the estimated free memory"):
        exec_jobs_dynamic(
            jobs=[lambda dev: None],
            devices=[torch.device("cuda:0")],
            max_workers=2,
            memory_estimates=[10_000_000_000],
        )


@patch(_PATCH_TARGET)
def test_single_worker_raises_when_job_exceeds_capacity(mock_mem_info):
    mock_mem_info.return_value = (1000, 96_000_000_000)
    with pytest.raises(RuntimeError, match="exceeds the estimated free memory"):
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
