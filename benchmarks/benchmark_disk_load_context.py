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


def run(
    num_tensors: int, num_reads: int | None = None, numel: int = 8
) -> tuple[float, float]:
    """Onload `num_reads` tensors (default: all) from a shard of `num_tensors`."""
    num_reads = num_tensors if num_reads is None else num_reads
    with tempfile.TemporaryDirectory() as directory:
        offload_dir = os.path.join(directory, "offload")
        os.mkdir(offload_dir)
        shard = os.path.join(directory, "model-00001-of-00001.safetensors")
        save_file({f"w{i}": torch.zeros(numel) for i in range(num_tensors)}, shard)

        DiskCache.index = {}
        offloaded = []
        for i in range(num_reads):
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

    return plain_time, grouped_time


def report(num_tensors: int) -> None:
    plain_time, grouped_time = run(num_tensors)
    print(
        f"  {num_tensors:7d}  {plain_time * 1e3:10.2f}  {grouped_time * 1e3:10.2f}  "
        f"{plain_time / grouped_time:7.1f}x"
    )


# One decoder layer's reads from one checkpoint file, taken from each model's
# model.safetensors.index.json. With plain transformers loading, MoE experts are
# fused on load and offloaded to their own files, so a layer reads only its
# attention, norm and router weights from the checkpoint. llm-compressor's
# `load_context` loads Qwen3-MoE and DeepSeek-V3 experts as per-expert 2D
# weights from the checkpoint instead. Layers that span files show the file
# they read most from.
LAYER_READS = (
    # (model and loading path, tensors in the file, tensors the layer reads)
    ("Llama-3.1-8B", 104, 9),
    ("Qwen3-8B", 114, 11),
    ("Qwen3-30B-A3B, experts fused", 1262, 9),
    ("Qwen3-235B-A22B, per-expert", 315, 235),
    ("DeepSeek-V3, per-expert", 586, 293),
    ("Qwen3-30B-A3B, per-expert", 1262, 393),
)


def report_layer_reads() -> None:
    print("\nOne decoder layer of real checkpoints, reads from one file")
    print(
        f"  {'model':>28}  {'tensors':>7}  {'reads':>5}  {'plain ms':>9}  "
        f"{'grouped ms':>10}  {'speedup':>7}"
    )
    for label, num_tensors, num_reads in LAYER_READS:
        plain_time, grouped_time = run(num_tensors, num_reads)
        print(
            f"  {label:>28}  {num_tensors:7d}  {num_reads:5d}  "
            f"{plain_time * 1e3:9.2f}  {grouped_time * 1e3:10.2f}  "
            f"{plain_time / grouped_time:6.1f}x"
        )


def main() -> None:
    print("DiskCache.onload of N tensors symlinked into one shard, warm page cache")
    print(f"median of {REPS}, torch {torch.__version__}\n")
    print(f"  {'tensors':>7}  {'plain ms':>10}  {'grouped ms':>10}  {'speedup':>8}")
    for num_tensors in (8, 32, 128, 384):
        report(num_tensors)
    print(
        "\nThe saving grows with the number of tensors in the shard, because each"
        "\nreopen re-parses a header whose size is proportional to that number."
    )
    report_layer_reads()


if __name__ == "__main__":
    main()
