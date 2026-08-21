# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    """
    n = len(jobs)

    # CPU path: run sequentially regardless of max_workers
    if all(d.type == "cpu" for d in devices):
        out = []
        for job in tqdm.tqdm(jobs, desc=desc):
            out.append(job(devices[0]))
        return out

    # Snapshot free VRAM once; all later decisions use accounting only
    initial_free = _snapshot_free(devices)

    # Single worker: pick the best device once upfront
    if max_workers == 1:
        device = max(initial_free, key=initial_free.get)
        out = []
        for i, job in enumerate(tqdm.tqdm(jobs, desc=desc)):
            if memory_estimates[i] > initial_free[device]:
                logger.warning(
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
