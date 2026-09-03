# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for fused vs unfused FP4 quantize+pack kernels.

Compares:
  - Fused: quantize_and_pack_fp4 (single kernel launch)
  - Unfused: quantize() + pack_fp4_to_uint8() (two kernel launches)
"""

import gc
import torch

from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    quantize_and_pack_fp4,
)
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationType,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 200  # CUDA events are accurate, don't need as many runs
GROUP_SIZE = 16  # NVFP4 always uses group_size=16


# Realistic weight shapes from popular LLMs
# Format: (name, rows, cols) - represents weight matrix shapes
MODEL_WEIGHT_SHAPES = [
    # Llama-7B (hidden=4096, intermediate=11008)
    ("Llama-7B MLP gate/up", 4096, 11008),
    ("Llama-7B MLP down", 11008, 4096),
    ("Llama-7B attn QKV", 4096, 4096 * 3),
    ("Llama-7B attn out", 4096, 4096),
    # Llama-13B (hidden=5120, intermediate=13824)
    ("Llama-13B MLP gate/up", 5120, 13824),
    ("Llama-13B MLP down", 13824, 5120),
    # Llama-70B (hidden=8192, intermediate=28672)
    ("Llama-70B MLP gate/up", 8192, 28672),
    ("Llama-70B MLP down", 28672, 8192),
    ("Llama-70B attn QKV", 8192, 8192 + 1024 * 2),  # GQA: 64 heads, 8 KV heads
    # Mixtral-8x7B (hidden=4096, intermediate=14336)
    ("Mixtral MLP gate/up", 4096, 14336),
    ("Mixtral MLP down", 14336, 4096),
    # Qwen2-72B (hidden=8192, intermediate=29568)
    ("Qwen2-72B MLP gate/up", 8192, 29568),
    ("Qwen2-72B MLP down", 29568, 8192),
]


def create_test_data(rows, cols, target_device):
    """Create test data and quantization parameters for FP4."""
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device=target_device)

    num_groups = cols // GROUP_SIZE
    scale = torch.rand(rows, num_groups, dtype=torch.bfloat16, device=target_device) + 0.1
    global_scale = torch.tensor(1.0, dtype=torch.float32, device=target_device)

    return x, scale, global_scale


def unfused_quantize_pack(x, scale, global_scale, args):
    """Unfused approach: separate quantize + pack kernels."""
    quantized = quantize(
        x=x,
        scale=scale,
        global_scale=global_scale,
        zero_point=None,
        args=args,
    )
    packed = pack_fp4_to_uint8(quantized)
    return packed


def fused_quantize_pack(x, scale, global_scale):
    """Fused approach: single kernel launch."""
    return quantize_and_pack_fp4(
        x=x,
        scale=scale,
        global_scale=global_scale,
        zero_point=None,
        group_size=GROUP_SIZE,
    )


def benchmark_kernel(func, *args, name="", warmup=False):
    """Benchmark a kernel function on CUDA using CUDA events for accurate timing."""
    if warmup:
        print(f"  Warming up {name}...")
        for _ in range(50):
            _ = func(*args)
        torch.cuda.synchronize()
        print("  Warmup complete, starting benchmark...")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    times = []

    for _ in range(N_RUNS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(*args)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)  # milliseconds
        times.append(elapsed_ms / 1000.0)  # convert to seconds

        del result

    # Use median for robustness against outliers
    times.sort()
    median_time = times[len(times) // 2]

    # Variance stats for debugging stability
    min_time = times[0]
    max_time = times[-1]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    variance_ratio = max_time / min_time if min_time > 0 else float('inf')

    print(f"    {name}: median={median_time*1000:.3f}ms, "
          f"min={min_time*1000:.3f}ms, max={max_time*1000:.3f}ms, "
          f"p10={p10*1000:.3f}ms, p90={p90*1000:.3f}ms, "
          f"variance_ratio={variance_ratio:.2f}x")

    return median_time


def run_benchmark(name, rows, cols):
    """Run benchmarks for a specific weight shape."""
    print(f"\n{'='*80}")
    print(f"{name}: {rows}x{cols} = {rows*cols/1e6:.1f}M elements")
    print("=" * 80)

    x, scale, global_scale = create_test_data(rows, cols, device)

    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        group_size=GROUP_SIZE,
        symmetric=True,
    )

    # Benchmark unfused (quantize + pack)
    print("\nRunning unfused (quantize + pack)...")
    time_unfused = benchmark_kernel(
        unfused_quantize_pack, x, scale, global_scale, args,
        name="unfused", warmup=True
    )

    # Benchmark fused
    print("\nRunning fused (quantize_and_pack_fp4)...")
    time_fused = benchmark_kernel(
        fused_quantize_pack, x, scale, global_scale,
        name="fused", warmup=True
    )

    # Verify correctness
    packed_unfused = unfused_quantize_pack(x.clone(), scale.clone(), global_scale.clone(), args)
    packed_fused = fused_quantize_pack(x.clone(), scale.clone(), global_scale.clone())
    correct = torch.equal(packed_unfused, packed_fused)

    if not correct:
        diff_count = (packed_unfused != packed_fused).sum().item()
        print(f"\nWarning: outputs differ at {diff_count} positions!")

    # Cleanup
    del x, scale, global_scale, packed_unfused, packed_fused
    torch.cuda.empty_cache()
    gc.collect()

    speedup = time_unfused / time_fused if time_fused > 0 else 0
    savings_ms = (time_unfused - time_fused) * 1000

    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "elements_M": rows * cols / 1e6,
        "unfused_ms": time_unfused * 1000,
        "fused_ms": time_fused * 1000,
        "speedup": speedup,
        "savings_ms": savings_ms,
        "correct": correct,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, benchmark requires GPU")
        return

    print("Benchmarking Fused vs Unfused FP4 Quantize+Pack")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"N_RUNS: {N_RUNS}")
    print(f"Group size: {GROUP_SIZE}")

    results = []

    for name, rows, cols in MODEL_WEIGHT_SHAPES:
        # Check if tensor fits in GPU memory (rough estimate: 2 bytes per element for bf16)
        # (Uses H100 with 80GB of memory as a reference.)
        mem_required_gb = (rows * cols * 2) / (1024**3)
        if mem_required_gb > 70:  # Leave headroom on 80GB H100
            print(f"\nSkipping {name} ({rows}x{cols}) - too large ({mem_required_gb:.1f}GB)")
            continue

        result = run_benchmark(name, rows, cols)
        results.append(result)

    # Print summary
    print("\n" + "=" * 120)
    print("SUMMARY: Fused vs Unfused FP4 Quantize+Pack")
    print("=" * 120)
    print(f"{'Model Weight':<25} {'Size':<18} {'Elements':<10} "
          f"{'Unfused (ms)':<14} {'Fused (ms)':<14} {'Speedup':<10} {'Savings':<12} {'Correct'}")
    print("-" * 120)

    total_unfused = 0
    total_fused = 0

    for r in results:
        size_str = f"{r['rows']}x{r['cols']}"
        correct_str = "Yes" if r["correct"] else "NO"
        print(f"{r['name']:<25} {size_str:<18} {r['elements_M']:>7.1f}M  "
              f"{r['unfused_ms']:>11.3f} ms  {r['fused_ms']:>11.3f} ms  "
              f"{r['speedup']:>6.2f}x    {r['savings_ms']:>+8.3f} ms  {correct_str}")
        total_unfused += r['unfused_ms']
        total_fused += r['fused_ms']

    print("-" * 120)
    total_speedup = total_unfused / total_fused if total_fused > 0 else 0
    total_savings = total_unfused - total_fused
    print(f"{'TOTAL':<25} {'':<18} {'':<10}  "
          f"{total_unfused:>11.3f} ms  {total_fused:>11.3f} ms  "
          f"{total_speedup:>6.2f}x    {total_savings:>+8.3f} ms")

    print("\nNote: 'Savings' shows time saved per weight tensor by using fused kernel.")
    print("For full model compression, multiply savings by number of weight tensors.")


if __name__ == "__main__":
    main()
