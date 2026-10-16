# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script comparing pack_to_int32 PyTorch vs Triton paths.

Compares:
- PyTorch baseline: Pure PyTorch scatter_add implementation on GPU
- Triton: pack_to_int32 on GPU (uses Triton kernel)

Tests both:
- packed_dim=0: Uses col-parallel kernel (no transpose)
- packed_dim=1: Uses row-parallel kernel (grouped packing)

Real-world scenarios:
- LLaMA-7B: weights like (4096, 4096), (11008, 4096), (4096, 11008)
- LLaMA-13B: weights like (5120, 5120), (13824, 5120)
- LLaMA-70B: weights like (8192, 8192), (28672, 8192)
- Falcon-7B: (4544, 4544), (4544, 18176)
- Zero-point shapes: (512, 1), (512, 8), (1024, 4)
"""

# Ensure we use the local development source, not installed package
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import gc
import math

import torch

from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 100


def _pack_to_int32_pytorch(value: torch.Tensor, num_bits: int, packed_dim: int = 1):
    """
    Pure PyTorch implementation of pack_to_int32 for benchmarking.

    This bypasses ImplBackend dispatch to always use PyTorch scatter_add,
    even on GPU, for fair comparison against Triton.
    """
    # Convert to unsigned range
    offset = 1 << (num_bits - 1)
    value = value.to(torch.int32) + offset

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    packed_cols = math.ceil(cols * num_bits / 32)

    # Pad to multiple of 32
    padded_cols = math.ceil(cols / 32) * 32
    if padded_cols > cols:
        value = torch.nn.functional.pad(value, (0, padded_cols - cols))

    num_groups = padded_cols // 32
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, 32)
    output_g = torch.zeros(rows_g, num_bits, dtype=torch.int32, device=value.device)

    elem_i = torch.arange(32, device=value.device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32

    output_g.scatter_add_(
        1,
        word_idx.unsqueeze(0).expand(rows_g, -1),
        value_g << bit_offset.unsqueeze(0),
    )

    ov = bit_offset + num_bits - 32
    ov_mask = ov > 0
    if ov_mask.any():
        ov_vals = value_g[:, ov_mask] >> (num_bits - ov[ov_mask]).unsqueeze(0)
        output_g.scatter_add_(
            1,
            (word_idx[ov_mask] + 1).unsqueeze(0).expand(rows_g, -1),
            ov_vals,
        )

    output = output_g.view(rows, num_groups * num_bits)[:, :packed_cols]

    if packed_dim == 0:
        output = output.transpose(0, 1)

    return output


# Real-world weight matrix shapes (rows, cols, description)
# These represent typical LLM weight matrices
WEIGHT_SHAPES = [
    # LLaMA-7B / Mistral-7B style
    (4096, 4096, "LLaMA-7B q/k/v/o_proj"),
    (11008, 4096, "LLaMA-7B gate/up_proj"),
    (4096, 11008, "LLaMA-7B down_proj"),
    # LLaMA-13B style
    (5120, 5120, "LLaMA-13B q/k/v/o_proj"),
    (13824, 5120, "LLaMA-13B gate/up_proj"),
    # LLaMA-70B style
    (8192, 8192, "LLaMA-70B q/k/v/o_proj"),
    (28672, 8192, "LLaMA-70B gate/up_proj"),
    # Falcon-7B style (non-power-of-2)
    (4544, 4544, "Falcon-7B dense"),
    (4544, 18176, "Falcon-7B dense_h_to_4h"),
]

# Zero-point shapes for asymmetric quantization
# packed_dim=0 is used for zero-points
ZERO_POINT_SHAPES = [
    (512, 1, "ZP per-tensor"),
    (512, 8, "ZP group-128"),
    (1024, 4, "ZP group-256"),
    (4096, 32, "ZP group-128 large"),
]


def create_test_data(rows, cols, num_bits, target_device):
    """Create random int8 quantized weights."""
    max_val = (1 << (num_bits - 1)) - 1
    min_val = -(1 << (num_bits - 1))
    x = torch.randint(
        min_val, max_val + 1, (rows, cols), dtype=torch.int8, device=target_device
    )
    return x


def benchmark_cuda(func, x, num_bits, packed_dim, name, warmup=False):
    """Benchmark a packing function on CUDA using CUDA events for accurate timing."""
    x = x.clone()

    # Warmup phase
    if warmup:
        for _ in range(20):
            _ = func(x, num_bits, packed_dim)
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    times = []

    for _ in range(N_RUNS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(x, num_bits, packed_dim)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)

    # Remove outliers (top/bottom 10%)
    times.sort()
    n_trim = max(1, len(times) // 10)
    trimmed_times = times[n_trim:-n_trim] if len(times) > 2 * n_trim else times

    avg_ms = sum(trimmed_times) / len(trimmed_times)
    min_ms = min(trimmed_times)
    max_ms = max(trimmed_times)

    return avg_ms, min_ms, max_ms, result


def verify_correctness(x, num_bits, packed_dim):
    """Verify that Triton implementation matches PyTorch baseline."""
    result_pytorch = _pack_to_int32_pytorch(x.clone(), num_bits, packed_dim)
    result_triton = pack_to_int32(x, num_bits, packed_dim)
    return torch.equal(result_pytorch, result_triton)


def run_benchmark_for_shape(rows, cols, num_bits, packed_dim, shape_name):
    """Run benchmark for a specific shape configuration."""
    x = create_test_data(rows, cols, num_bits, device)

    # Verify correctness
    correct = verify_correctness(x, num_bits, packed_dim)

    # Benchmark PyTorch baseline on GPU
    avg_orig, _, _, _ = benchmark_cuda(
        _pack_to_int32_pytorch, x, num_bits, packed_dim, "PyTorch", warmup=True
    )

    # Benchmark Triton on GPU
    avg_accel, _, _, _ = benchmark_cuda(
        pack_to_int32, x, num_bits, packed_dim, "Triton", warmup=True
    )

    # Calculate speedup
    speedup = avg_orig / avg_accel if avg_accel > 0 else float("inf")

    return {
        "shape": (rows, cols),
        "shape_name": shape_name,
        "num_bits": num_bits,
        "packed_dim": packed_dim,
        "pytorch_ms": avg_orig,
        "triton_ms": avg_accel,
        "speedup": speedup,
        "correct": correct,
    }


def print_results_table(results, title):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(
        f"{'Shape':<15} {'Name':<25} {'Bits':>4} {'Dim':>4} "
        f"{'PyTorch (ms)':>12} {'Triton (ms)':>12} "
        f"{'Speedup':>10} {'Correct':>8}"
    )
    print("-" * 100)

    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        correct_str = "✓" if r["correct"] else "✗"
        print(
            f"{shape_str:<15} {r['shape_name']:<25} {r['num_bits']:>4} {r['packed_dim']:>4} "
            f"{r['pytorch_ms']:>12.3f} {r['triton_ms']:>12.3f} "
            f"{r['speedup']:>9.2f}x {correct_str:>8}"
        )

    print("=" * 100)


def print_summary(results_dim0, results_dim1):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results_dim0:
        avg_speedup_dim0 = sum(r["speedup"] for r in results_dim0) / len(results_dim0)
        print(f"\npacked_dim=0 (col-parallel kernel, no transpose):")
        print(f"  Average speedup: {avg_speedup_dim0:.2f}x")
        print(f"  All correct: {'✓' if all(r['correct'] for r in results_dim0) else '✗'}")

    if results_dim1:
        avg_speedup_dim1 = sum(r["speedup"] for r in results_dim1) / len(results_dim1)
        print(f"\npacked_dim=1 (row-parallel kernel, grouped packing):")
        print(f"  Average speedup: {avg_speedup_dim1:.2f}x")
        print(f"  All correct: {'✓' if all(r['correct'] for r in results_dim1) else '✗'}")

    print("=" * 60)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, benchmark requires GPU")
        return

    from compressed_tensors.utils.triton import HAS_TRITON

    if not HAS_TRITON:
        print("Triton is not available, skipping benchmark")
        return

    print("=" * 60)
    print("pack_to_int32 Triton Kernel Benchmark")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"N_RUNS: {N_RUNS}")
    print("\nKernel paths:")
    print("  - packed_dim=0: col-parallel kernel (no transpose)")
    print("  - packed_dim=1: row-parallel kernel (grouped packing)")

    results_dim0 = []
    results_dim1 = []

    # Test bit depths
    bit_depths = [4, 8]

    # ==========================================================================
    # Benchmark packed_dim=1 (row-parallel kernel) - Weight packing
    # ==========================================================================
    print("\n" + "#" * 80)
    print("# packed_dim=1: Row-parallel kernel (weights)")
    print("#" * 80)

    for num_bits in bit_depths:
        print(f"\n--- {num_bits}-bit quantization ---")
        for rows, cols, name in WEIGHT_SHAPES:
            result = run_benchmark_for_shape(rows, cols, num_bits, packed_dim=1, shape_name=name)
            results_dim1.append(result)
            print(
                f"  {name:<30} {rows}x{cols}: "
                f"PyTorch={result['pytorch_ms']:.3f}ms, "
                f"Triton={result['triton_ms']:.3f}ms, "
                f"Speedup={result['speedup']:.2f}x"
            )

    # ==========================================================================
    # Benchmark packed_dim=0 (col-parallel kernel) - Weight & ZP packing
    # ==========================================================================
    print("\n" + "#" * 80)
    print("# packed_dim=0: Col-parallel kernel (weights & zero-points)")
    print("#" * 80)

    for num_bits in bit_depths:
        print(f"\n--- {num_bits}-bit quantization (weights) ---")
        for rows, cols, name in WEIGHT_SHAPES:
            result = run_benchmark_for_shape(rows, cols, num_bits, packed_dim=0, shape_name=name)
            results_dim0.append(result)
            print(
                f"  {name:<30} {rows}x{cols}: "
                f"PyTorch={result['pytorch_ms']:.3f}ms, "
                f"Triton={result['triton_ms']:.3f}ms, "
                f"Speedup={result['speedup']:.2f}x"
            )

    for num_bits in bit_depths:
        print(f"\n--- {num_bits}-bit quantization (zero-points) ---")
        for rows, cols, name in ZERO_POINT_SHAPES:
            result = run_benchmark_for_shape(rows, cols, num_bits, packed_dim=0, shape_name=name)
            results_dim0.append(result)
            print(
                f"  {name:<30} {rows}x{cols}: "
                f"PyTorch={result['pytorch_ms']:.3f}ms, "
                f"Triton={result['triton_ms']:.3f}ms, "
                f"Speedup={result['speedup']:.2f}x"
            )

    # ==========================================================================
    # Print summary tables
    # ==========================================================================
    print_results_table(results_dim1, "RESULTS: packed_dim=1 (row-parallel kernel)")
    print_results_table(results_dim0, "RESULTS: packed_dim=0 (col-parallel kernel)")
    print_summary(results_dim0, results_dim1)


if __name__ == "__main__":
    main()
