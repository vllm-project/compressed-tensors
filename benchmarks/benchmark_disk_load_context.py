# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Measure what `disk_load_context` saves when onloading many tensors from one shard.

The benefit only exists on multi-tensor safetensors shards. Those reach
`DiskCache` through `create_checkpoint_symlink`, which gives every tensor its own
symlink into the shared shard, so that is the path measured here: real
`DiskCache.onload` calls through per-tensor symlinks.

`safe_open` parses a header describing every tensor in the file, so reopening
the shard for each of N tensors is quadratic in N. Holding one handle makes the
header parse happen once. What remains per read (index lookup, `to_tensor`, the
dtype cast, resolving the symlink) is not affected, which is why the end-to-end
ratio is smaller than the open-versus-reuse ratio alone.

Run:
    python benchmarks/benchmark_disk_load_context.py
"""

import gc
import os
import statistics
import tempfile
import time

import torch
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.disk_utils import disk_load_context
from safetensors.torch import save_file


REPS = 9


def _median(fn) -> float:
    times = []
    for _ in range(REPS):
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run(num_tensors: int, numel: int = 8) -> None:
    with tempfile.TemporaryDirectory() as directory:
        offload_dir = os.path.join(directory, "offload")
        os.mkdir(offload_dir)
        shard = os.path.join(directory, "model-00001-of-00001.safetensors")
        save_file({f"w{i}": torch.zeros(numel) for i in range(num_tensors)}, shard)

        DiskCache.index = {}
        offloaded = []
        for i in range(num_tensors):
            meta = torch.empty(numel, device="meta")
            DiskCache.create_checkpoint_symlink(
                meta,
                {"safetensors_file": shard, "weight_name": f"w{i}", "dtype": "float32"},
                offload_dir,
            )
            offloaded.append(meta)
        cache = DiskCache("cpu", offload_dir=offload_dir)

        def plain():
            for meta in offloaded:
                cache.onload(meta)

        def grouped():
            with disk_load_context():
                for meta in offloaded:
                    cache.onload(meta)

        plain()  # warm the page cache
        plain_time = _median(plain)
        grouped_time = _median(grouped)

    print(
        f"  {num_tensors:7d}  {plain_time * 1e3:10.2f}  {grouped_time * 1e3:10.2f}  "
        f"{plain_time / grouped_time:7.1f}x"
    )


def main() -> None:
    print("DiskCache.onload of N tensors symlinked into one shard, warm page cache")
    print(f"median of {REPS}, torch {torch.__version__}\n")
    print(f"  {'tensors':>7}  {'plain ms':>10}  {'grouped ms':>10}  {'speedup':>8}")
    for num_tensors in (8, 32, 128, 384):
        run(num_tensors)
    print(
        "\nThe saving grows with the number of tensors in the shard, because each"
        "\nreopen re-parses a header whose size is proportional to that number."
    )


if __name__ == "__main__":
    main()
