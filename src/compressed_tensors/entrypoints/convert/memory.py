# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import sys
import traceback as tb
import weakref
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from functools import partial
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

import psutil
import torch
import tqdm
from compressed_tensors.utils.safetensors_load import (
    InverseWeightMap,
    load_tensors_from_inverse_weight_map,
)
from loguru import logger
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


if TYPE_CHECKING:
    from compressed_tensors.entrypoints.convert.converters import Converter


__all__ = ["exec_jobs_dynamic", "estimate_job_memory", "TensorProfiler"]


_FALLBACK_MULTIPLIER = 2.5


class MemoryProfile:
    _timelines: dict[torch.device, list[int]]

    def __init__(self):
        self._timelines: dict[torch.device, list[int]] = dict()

    def add(self, device: torch.device, size: int):
        if device not in self._timelines:
            timeline = [0 for _ in range(max(len(self), 1))]
            self._timelines[device] = timeline

        for dev in self._timelines:
            if dev == device:
                diff = size
            else:
                diff = 0

            self._timelines[dev].append(self._timelines[dev][-1] + diff)

    def subtract(self, device: torch.device, size: int):
        self.add(device, -size)

    @property
    def current(self) -> dict[torch.device, int]:
        return {device: self._timelines[device][-1] for device in self._timelines}

    @property
    def peak(self) -> dict[torch.device, int]:
        return {device: max(self._timelines[device]) for device in self._timelines}

    def __len__(self) -> int:
        return max((len(timeline) for timeline in self._timelines.values()), default=0)


class TensorProfiler(TorchDispatchMode):
    _tracked: set[int]
    _memory: MemoryProfile
    _exception: BaseException | None

    def __init__(self, catch_exception: bool = True):
        self._tracked = set()
        self._memory = MemoryProfile()
        self._catch_exception = catch_exception
        self._exception = None

    # ::::::::::::::::::::::::::::::::::::::::::::::::
    # 📤 Public API — user-facing methods
    # ::::::::::::::::::::::::::::::::::::::::::::::::

    @property
    def memory(self) -> dict[torch.device | str, int]:
        ret = self._memory.current.copy()
        total = sum(ret.values(), start=0)
        ret.update({"total": total})
        return ret

    @property
    def memory_peak(self) -> dict[torch.device | str, int]:
        ret = self._memory.peak.copy()
        all = max(ret.values(), default=0)
        ret.update({"all": all})
        return ret

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    # ::::::::::::::::::::::::::::::::::::::::::::::::
    # ⚙️ Tracking - Dispatch overload and finalizers
    # ::::::::::::::::::::::::::::::::::::::::::::::::

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        ret = func(*args, **(kwargs or {}))

        for obj in tree_leaves(ret):
            if isinstance(obj, torch.Tensor):
                self._track(obj.untyped_storage())

        return ret

    def _track(self, storage: torch.UntypedStorage):
        hash = storage._cdata
        size = storage.nbytes()
        device = storage.device

        # skip if already tracked
        if hash in self._tracked:
            return

        # track
        self._memory.add(device, size)
        self._tracked.add(hash)

        # register finalizer to subtract memory
        finalizer = partial(self._untrack, hash, size, device)
        weakref.finalize(storage, finalizer)  # triggers regardless of gc

    def _untrack(self, hash: int, size: int, device: torch.device):
        # skip if no longer tracking
        if hash not in self._tracked:
            return

        # untrack
        self._memory.subtract(device, size)
        self._tracked.remove(hash)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        if self._catch_exception and exc_type is not None:
            self._exception = exc_value
            tb.print_exception(exc_type, exc_value, traceback, file=sys.stderr)

        self._tracked = set()
        return super().__exit__(exc_type, exc_value, traceback) or self._catch_exception


def estimate_job_memory(
    inverse_weight_map: InverseWeightMap,
    converters: list["Converter"],
) -> int:
    """
    Estimate the peak device memory (in bytes) for a single conversion job by
    profiling it on meta tensors.

    Loads the job's tensors on the meta device and runs each converter's
    meta-safe ``validate`` under a :class:`TensorProfiler`, returning the peak
    memory observed on the meta device. If profiling fails, falls back to a
    multiple of the input tensor footprint. No real device memory is allocated.

    :param inverse_weight_map: mapping of source file path -> tensor names,
        identifying every weight (and cross-shard dependency) the job loads
    :param converters: converters applied in order; used to simulate the
        converted output size on meta tensors
    :returns: estimated peak memory in bytes
    """
    tensors: dict[str, torch.Tensor] = {}
    with TensorProfiler() as prof:
        tensors = load_tensors_from_inverse_weight_map(
            inverse_weight_map, device="meta"
        )
        for converter in converters:
            tensors = converter.validate(tensors)

    meta = torch.device("meta")
    if prof.exception is not None or meta not in prof.memory_peak:
        fallback = int(
            sum(
                tensor.nbytes
                for tensor in tensors.values()
                if isinstance(tensor, torch.Tensor)
            )
            * _FALLBACK_MULTIPLIER
        )
        logger.warning(
            "Failed to profile conversion memory usage. Falling back to "
            f"{_FALLBACK_MULTIPLIER}x size of tensor inputs "
            f"({fallback / 1e9:.2f} GB)."
        )
        return fallback

    return prof.memory_peak[meta]


# mountinfo escapes space, tab, newline and backslash in paths as octal
_MOUNTINFO_ESCAPE = re.compile(r"\\(040|011|012|134)")

# limit file, usage file and reclaimable `memory.stat` key of each cgroup version
_V1_MEMORY_FILES = (
    "memory.limit_in_bytes",
    "memory.usage_in_bytes",
    "total_inactive_file",
)
_V2_MEMORY_FILES = ("memory.max", "memory.current", "inactive_file")


def _unescape_mountinfo(field: str) -> str:
    return _MOUNTINFO_ESCAPE.sub(lambda match: chr(int(match[1], 8)), field)


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _read_stat(path: Path, key: str) -> int:
    try:
        for line in path.read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except (OSError, ValueError):
        pass
    return 0


def _cgroup_available_bytes(
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    mountinfo: Path = Path("/proc/self/mountinfo"),
) -> Optional[int]:
    """
    Estimated memory this process can still allocate under the cgroup memory limits
    it can read: the smallest `limit - (usage - inactive file cache)` over the
    process's cgroup and its ancestors, as seen through every mount of the cgroup
    hierarchy. Supports cgroup v2 and the v1 memory controller. Ancestors outside
    the mounts (e.g. above a container's cgroup namespace) cannot be read, so their
    limits and their other members' usage are not taken into account, and a cgroup
    outside this process's cgroup namespace (a path or mount root with "..") is not
    resolved.

    :return: available bytes, or None if no cgroup memory limit could be read
    """
    try:
        cgroup_lines = proc_cgroup.read_text().splitlines()
        mount_lines = mountinfo.read_text().splitlines()
    except OSError:
        return None

    # each cgroup line is "hierarchy-id:controller-list:path"
    entries = [line.split(":", 2) for line in cgroup_lines if line.count(":") >= 2]
    # each mountinfo line is "id parent-id major:minor root mount-point options
    # [optional fields...] - fs-type source super-options"
    mounts = []
    for line in mount_lines:
        fields, _, fs_fields = line.partition(" - ")
        fields, fs_fields = fields.split(), fs_fields.split()
        if len(fields) >= 5 and len(fs_fields) >= 3:
            root, point = _unescape_mountinfo(fields[3]), _unescape_mountinfo(fields[4])
            mounts.append((fs_fields[0], fs_fields[2].split(","), root, point))

    v1 = [
        path for _, controllers, path in entries if "memory" in controllers.split(",")
    ]
    v2 = [path for hierarchy, _, path in entries if hierarchy == "0"]
    if v1:
        path, files = v1[0], _V1_MEMORY_FILES
        candidates = [
            (root, point)
            for fs_type, options, root, point in mounts
            if fs_type == "cgroup" and "memory" in options
        ]
    elif v2:
        path, files = v2[0], _V2_MEMORY_FILES
        candidates = [
            (root, point) for fs_type, _, root, point in mounts if fs_type == "cgroup2"
        ]
    else:
        return None

    # cgroups outside this process's cgroup namespace are shown with ".."
    if ".." in PurePosixPath(path).parts:
        return None

    # a mount exposes the hierarchy below its root, so resolve the process's cgroup
    # relative to that root. Mounts may expose different parts of the hierarchy or
    # hide some files, so take the smallest headroom over all of them
    available = None
    for root, point in candidates:
        if ".." in PurePosixPath(root).parts:
            continue
        try:
            relative = PurePosixPath(path).relative_to(root)
        except ValueError:
            continue
        mount_point = Path(point)
        headroom = _cgroup_headroom(mount_point / relative, mount_point, *files)
        if headroom is not None:
            available = headroom if available is None else min(available, headroom)

    return available


def _cgroup_headroom(
    leaf: Path, mount_point: Path, limit_file: str, usage_file: str, inactive_key: str
) -> Optional[int]:
    """Smallest headroom over `leaf` and its ancestors up to `mount_point`"""
    available = None
    for directory in (leaf, *leaf.parents):
        # cgroup v2 reports "max" when there is no limit, which is skipped here
        limit = _read_int(directory / limit_file)
        usage = _read_int(directory / usage_file)
        if limit is not None and usage is not None:
            # page cache counts as usage, but inactive file pages are reclaimable
            inactive = _read_stat(directory / "memory.stat", inactive_key)
            headroom = max(0, limit - max(0, usage - inactive))
            available = headroom if available is None else min(available, headroom)
        if directory == mount_point:
            break

    return available


def _host_available_bytes() -> int:
    """
    Estimated host memory available to this process: the system's available memory,
    capped by the cgroup memory limits this process can read (see
    `_cgroup_available_bytes`). Inside a container, psutil alone reports the memory
    of the whole host rather than the container's limit.
    """
    available = psutil.virtual_memory().available
    cgroup_available = _cgroup_available_bytes()
    if cgroup_available is not None:
        available = min(available, cgroup_available)
    return available


def _snapshot_free(devices: list[torch.device]) -> dict[torch.device, int]:
    """
    Query free memory once per device. Accelerators report free device memory.
    When every device is the cpu, the cpu reports the host memory available to this
    process; otherwise cpu devices are skipped, so that host memory does not draw
    jobs away from the accelerators.
    """
    cpu_only = all(d.type == "cpu" for d in devices)
    free = {}
    for d in devices:
        if d in free:
            continue
        if d.type != "cpu":
            mem_free, _ = torch.accelerator.memory.get_memory_info(d)
            free[d] = mem_free
        elif cpu_only:
            free[d] = _host_available_bytes()
    return free


def _free_bytes(
    dev: torch.device,
    initial_free: dict[torch.device, int],
    reserved: dict[torch.device, int],
) -> int:
    """Available memory for *dev*: initial snapshot minus in-flight reservations.

    Devices missing from *initial_free* (cpu devices alongside accelerators, see
    ``_snapshot_free``) return 0 and are never picked by ``_pick_device``.
    """
    return max(0, initial_free.get(dev, 0) - reserved.get(dev, 0))


def _pick_device(
    devices: list[torch.device],
    required: int,
    initial_free: dict[torch.device, int],
    reserved: dict[torch.device, int],
) -> torch.device | None:
    """Return the device with the most available memory that can fit *required*
    bytes, or ``None`` if nothing qualifies."""
    best, best_free = None, -1
    for dev in devices:
        available = _free_bytes(dev, initial_free, reserved)
        if available >= required and available > best_free:
            best, best_free = dev, available
    return best


def _run_job_on_device(job: Callable[[torch.device], Any], device: torch.device) -> Any:
    """Run a job with the worker thread's assigned accelerator device selected.

    Accelerator current-device state is thread-local. Passing ``cuda:N`` to a
    job is not enough for kernels that validate pointers against the current
    device. Keep allocation and kernel launch in the same device context for
    the full job. CPU jobs run without a device context.
    """

    context = (
        torch.accelerator.device_index(device.index)
        if device.type != "cpu"
        else nullcontext()
    )
    with context:
        return job(device)


def exec_jobs_dynamic(
    jobs: list[Callable[[torch.device], Any]],
    devices: list[torch.device],
    max_workers: int,
    memory_estimates: list[int],
    desc: str = "Processing",
) -> list:
    """Run *jobs* across *devices*, assigning each job at submit time to
    whichever device has the most free memory.

    Each job is a callable that accepts a single ``torch.device`` argument and
    returns its result. Free memory is queried once at startup (host memory for
    cpu-only runs, see ``_snapshot_free``); subsequent scheduling decisions rely on
    reservation accounting so we never re-query in a hot loop. Reservations track
    the estimates of this call's in-flight jobs against that initial snapshot, so
    memory used or freed by anything else afterwards is not seen. Effective
    concurrency is capped by estimated capacity: even if ``max_workers`` is high,
    jobs are held back until a device can actually fit the estimated footprint.
    All cpu devices share the same host memory, so they are scheduled as one device.

    :param jobs: list of callables, each accepting a device and returning a result
    :param devices: list of devices to schedule across
    :param max_workers: upper bound on concurrent workers
    :param memory_estimates: per-job memory estimate in bytes, parallel to *jobs*
    :param desc: tqdm progress bar label
    :return: list of results in the same order as *jobs*
    :raises ValueError: if inputs are invalid (length mismatch, negative estimates,
        max_workers < 1, or empty devices with non-empty jobs)
    :raises RuntimeError: if a job's estimate exceeds the free memory of every
        device, before any job runs. Note: if a worker raises mid-run, the
        ThreadPoolExecutor drains all in-flight jobs before the exception surfaces
        to the caller.
    """
    n = len(jobs)

    if len(memory_estimates) != n:
        raise ValueError(
            f"memory_estimates length ({len(memory_estimates)}) must match "
            f"jobs length ({n})"
        )
    if any(e < 0 for e in memory_estimates):
        raise ValueError("memory_estimates must not contain negative values")
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")
    if n > 0 and not devices:
        raise ValueError("devices must not be empty when jobs are provided")
    if n == 0:
        return []

    # cpu aliases (e.g. "cpu" and "cpu:0") draw on one host memory budget
    devices = [torch.device("cpu") if d.type == "cpu" else d for d in devices]

    # Snapshot free memory once; all later decisions use accounting only
    initial_free = _snapshot_free(devices)
    if not initial_free:
        raise RuntimeError(
            "Could not query free memory for any device. "
            "Ensure at least one non-CPU device is accessible."
        )

    # Reject jobs that cannot fit on any device before converting anything
    capacity = max(initial_free.values())
    for i, estimate in enumerate(memory_estimates):
        if estimate > capacity:
            raise RuntimeError(
                f"Job {i} needs an estimated {estimate / 1e9:.2f} GB, which exceeds "
                f"the estimated free memory of every device (at most "
                f"{capacity / 1e9:.2f} GB)"
            )

    # Single worker: pick the best device once upfront
    if max_workers == 1:
        device = max(initial_free, key=initial_free.get)
        return [_run_job_on_device(job, device) for job in tqdm.tqdm(jobs, desc=desc)]

    # Multi-worker: main thread schedules, workers execute
    reserved = {d: 0 for d in devices}
    results = [None] * n
    pending = list(range(n))
    fut_device: dict = {}

    with (
        tqdm.tqdm(total=n, desc=desc) as bar,
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        inflight: dict = {}

        while pending or inflight:
            for idx in list(pending):
                if len(inflight) >= max_workers:
                    break
                dev = _pick_device(
                    devices,
                    memory_estimates[idx],
                    initial_free,
                    reserved,
                )
                if dev is None:
                    continue

                fut = pool.submit(_run_job_on_device, jobs[idx], dev)
                inflight[fut] = idx
                fut_device[fut] = dev
                reserved[dev] += memory_estimates[idx]
                pending.remove(idx)
                logger.debug(
                    f"Job {idx} -> {dev} (~{memory_estimates[idx] / 1e9:.2f} GB)"
                )

            if not inflight:
                if not pending:
                    break
                # unreachable while every job passes the up-front check (with
                # nothing in flight, nothing is reserved); guards against waiting
                # on no futures in a busy loop if scheduling changes
                raise RuntimeError(
                    "No device has enough estimated free memory for any "
                    "remaining job"
                )

            done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)

            for f in done:
                i = inflight.pop(f)
                dev = fut_device.pop(f)
                reserved[dev] -= memory_estimates[i]
                results[i] = f.result()
                bar.update(1)

    return results
