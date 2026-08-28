# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from typing import Any

import torch
import tqdm
from loguru import logger


__all__ = ["exec_jobs_dynamic"]


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
            out.append(_run_job_on_device(job, devices[0]))
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
            out.append(_run_job_on_device(job, device))
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
