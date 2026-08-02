# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for offloading cache implementations.

Non-distributed caches (CPUCache, DiskCache) are always benchmarked.
Distributed caches (DistributedCPUCache, DistributedDiskCache) require
torch.distributed to be initialized. Run with torchrun to include them:

    torchrun --nproc_per_node 1 benchmarks/benchmark_offload_cache.py

Set CT_USE_INSTANT_TENSOR=1 to benchmark DiskCache with instanttensor:

    CT_USE_INSTANT_TENSOR=1 python benchmarks/benchmark_offload_cache.py
"""

import os
import sys
import tempfile
import time
import warnings
from typing import Optional, Type

import torch
import torch.distributed as dist

from compressed_tensors.offload.cache import (
    CPUCache,
    DiskCache,
    DistributedCPUCache,
    DistributedDiskCache,
    OffloadCache,
)


SIZE_LABELS = ["1e0", "1e3", "1e6", "1e9"]
SIZES = [int(float(s)) for s in SIZE_LABELS]
N_TRIALS = 3


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def make_cache(
    cache_cls: Type[OffloadCache],
    onload_device: str,
    offload_dir: Optional[str] = None,
) -> OffloadCache:
    kwargs: dict = {"onload_device": onload_device}
    if issubclass(cache_cls, DiskCache):
        kwargs["offload_dir"] = offload_dir
    return cache_cls(**kwargs)


def benchmark_allocate(
    cache_cls: Type[OffloadCache],
    tensor: torch.Tensor,
    onload_device: str,
    offload_dir: Optional[str],
    n_trials: int,
) -> float:
    times = []
    for _ in range(n_trials):
        cache = make_cache(cache_cls, onload_device, offload_dir)
        start = time.perf_counter()
        cache["key"] = tensor
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del cache["key"]
    return min(times)


def benchmark_onload(
    cache_cls: Type[OffloadCache],
    tensor: torch.Tensor,
    onload_device: str,
    offload_dir: Optional[str],
    n_trials: int,
) -> float:
    cache = make_cache(cache_cls, onload_device, offload_dir)
    cache["key"] = tensor
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = cache["key"]
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    del cache["key"]
    return min(times)


def benchmark_update(
    cache_cls: Type[OffloadCache],
    tensor: torch.Tensor,
    onload_device: str,
    offload_dir: Optional[str],
    n_trials: int,
) -> float:
    cache = make_cache(cache_cls, onload_device, offload_dir)
    cache["key"] = tensor
    new_tensor = torch.zeros_like(tensor)
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        cache["key"] = new_tensor
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    del cache["key"]
    return min(times)


def benchmark_delete(
    cache_cls: Type[OffloadCache],
    tensor: torch.Tensor,
    onload_device: str,
    offload_dir: Optional[str],
    n_trials: int,
) -> float:
    times = []
    for _ in range(n_trials):
        cache = make_cache(cache_cls, onload_device, offload_dir)
        cache["key"] = tensor
        start = time.perf_counter()
        del cache["key"]
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return min(times)


def run_benchmarks() -> list[tuple]:
    distributed = dist.is_available() and dist.is_initialized()

    onload_devices = ["cpu"]
    if torch.cuda.is_available():
        # pick the GPU with the most free memory
        best_gpu = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
        onload_devices.append(f"cuda:{best_gpu}")

    cache_classes: list[tuple[str, Type[OffloadCache]]] = [
        ("CPUCache", CPUCache),
        ("DiskCache", DiskCache),
    ]
    if distributed:
        cache_classes = [
            ("CPUCache", CPUCache),
            ("DistributedCPUCache", DistributedCPUCache),
            ("DiskCache", DiskCache),
            ("DistributedDiskCache", DistributedDiskCache),
        ]

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for cache_name, cache_cls in cache_classes:
            for onload_device in onload_devices:
                if cache_cls is CPUCache and onload_device != "cpu":
                    continue

                for size, size_label in zip(SIZES, SIZE_LABELS):
                    # skip sizes that won't fit in the target device's memory
                    if onload_device.startswith("cuda"):
                        dev_idx = int(onload_device.split(":")[-1]) if ":" in onload_device else 0
                        free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
                        tensor_bytes = size * 4  # float32
                        if tensor_bytes > free_bytes * 0.8:
                            continue

                    n_trials = 1 if size >= int(1e9) else N_TRIALS
                    tensor = torch.ones(size)
                    t_alloc = benchmark_allocate(
                        cache_cls, tensor, onload_device, tmpdir, n_trials
                    )
                    t_onload = benchmark_onload(
                        cache_cls, tensor, onload_device, tmpdir, n_trials
                    )
                    t_update = benchmark_update(
                        cache_cls, tensor, onload_device, tmpdir, n_trials
                    )
                    t_delete = benchmark_delete(
                        cache_cls, tensor, onload_device, tmpdir, n_trials
                    )
                    label = (
                        f"{cache_name}[{onload_device}]"
                        if onload_device != "cpu"
                        else cache_name
                    )
                    results.append(
                        (label, size_label, t_alloc, t_onload, t_update, t_delete)
                    )

    return results


def print_table(results: list[tuple]) -> None:
    col_widths = (30, 12, 16, 16, 16, 12)
    headers = (
        "Cache Type",
        "Size",
        "Allocate (s)",
        "Onload (s)",
        "Update (s)",
        "Delete (s)",
    )
    header = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header)
    print("-" * len(header))
    for cache_name, size_label, t_alloc, t_onload, t_update, t_delete in results:
        row = (
            cache_name.ljust(col_widths[0])
            + size_label.ljust(col_widths[1])
            + f"{t_alloc:.6f}".ljust(col_widths[2])
            + f"{t_onload:.6f}".ljust(col_widths[3])
            + f"{t_update:.6f}".ljust(col_widths[4])
            + f"{t_delete:.6f}".ljust(col_widths[5])
        )
        print(row)


if __name__ == "__main__":
    if "TORCHELASTIC_RUN_ID" in os.environ:
        from compressed_tensors.distributed import init_dist

        init_dist()

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    use_instant_tensor = bool(os.environ.get("CT_USE_INSTANT_TENSOR"))
    results = run_benchmarks()

    if is_main_process():
        loader = "instanttensor" if use_instant_tensor else "safetensors"
        print(f"DiskCache CUDA loader: {loader}")
        print()
        print_table(results)
