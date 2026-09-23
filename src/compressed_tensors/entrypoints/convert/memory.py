# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import traceback as tb
import weakref
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

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


def _snapshot_free(devices: list[torch.device]) -> dict[torch.device, int]:
    """Query free VRAM once per device. CPU devices are skipped."""
    free = {}
    for d in devices:
        if d.type != "cpu" and d not in free:
            mem_free, _ = torch.accelerator.memory.get_memory_info(d)
            free[d] = mem_free
    return free


def _free_bytes(
    dev: torch.device,
    initial_free: dict[torch.device, int],
    reserved: dict[torch.device, int],
) -> int:
    """Available VRAM for *dev*: initial snapshot minus in-flight reservations.

    CPU devices are not present in *initial_free* (skipped by
    ``_snapshot_free``), so they always return 0 and are never picked by
    ``_pick_device``. The CPU-only path is handled separately in
    ``exec_jobs_dynamic`` before any scheduling logic runs.
    """
    return max(0, initial_free.get(dev, 0) - reserved.get(dev, 0))


def _pick_device(
    devices: list[torch.device],
    required: int,
    initial_free: dict[torch.device, int],
    reserved: dict[torch.device, int],
) -> torch.device | None:
    """Return the device with the most available VRAM that can fit *required*
    bytes, or ``None`` if nothing qualifies."""
    best, best_free = None, -1
    for dev in devices:
        available = _free_bytes(dev, initial_free, reserved)
        if available >= required and available > best_free:
            best, best_free = dev, available
    return best


def exec_jobs_dynamic(
    jobs: list[Callable[[torch.device], Any]],
    devices: list[torch.device],
    max_workers: int,
    memory_estimates: list[int],
    desc: str = "Processing",
) -> list:
    """Run *jobs* across *devices*, assigning each job at submit time to
    whichever GPU has the most free memory.

    Each job is a callable that accepts a single ``torch.device`` argument and
    returns its result. Free VRAM is queried once at startup; subsequent
    scheduling decisions rely on reservation accounting so we never re-query
    the driver in a hot loop. Effective concurrency is capped by estimated GPU
    capacity: even if ``max_workers`` is high, jobs are held back until a GPU
    can actually fit the estimated footprint.

    :param jobs: list of callables, each accepting a device and returning a result
    :param devices: list of devices to schedule across
    :param max_workers: upper bound on concurrent workers
    :param memory_estimates: per-job memory estimate in bytes, parallel to *jobs*
    :param desc: tqdm progress bar label
    :return: list of results in the same order as *jobs*
    :raises ValueError: if inputs are invalid (length mismatch, negative estimates,
        max_workers < 1, or empty devices with non-empty jobs)
    :raises RuntimeError: if no device has enough estimated free memory for a job.
        Note: if a worker raises mid-run, the ThreadPoolExecutor drains all
        in-flight jobs before the exception surfaces to the caller.
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

    # CPU path: run sequentially regardless of max_workers
    if all(d.type == "cpu" for d in devices):
        out = []
        for job in tqdm.tqdm(jobs, desc=desc):
            out.append(job(devices[0]))
        return out

    # Snapshot free VRAM once; all later decisions use accounting only
    initial_free = _snapshot_free(devices)
    if not initial_free:
        raise RuntimeError(
            "Could not query free memory for any device. "
            "Ensure at least one non-CPU device is accessible."
        )

    # Single worker: pick the best device once upfront
    if max_workers == 1:
        device = max(initial_free, key=initial_free.get)
        out = []
        for i, job in enumerate(tqdm.tqdm(jobs, desc=desc)):
            if memory_estimates[i] > initial_free[device]:
                raise RuntimeError(
                    f"Job {i} (~{memory_estimates[i] / 1e9:.2f} GB) "
                    f"exceeds estimated capacity of {device}"
                )
            out.append(job(device))
        return out

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

                fut = pool.submit(jobs[idx], dev)
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
                raise RuntimeError(
                    "No device has enough estimated free memory for any "
                    "remaining job. Consider reducing max_workers or "
                    "increasing the memory estimate multiplier."
                )

            done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)

            for f in done:
                i = inflight.pop(f)
                dev = fut_device.pop(f)
                reserved[dev] -= memory_estimates[i]
                results[i] = f.result()
                bar.update(1)

    return results
