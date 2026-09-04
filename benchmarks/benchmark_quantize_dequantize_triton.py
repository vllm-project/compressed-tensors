# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for fused _quantize_dequantize Triton implementation.

Compares:
- Fused Triton quantize+dequantize (single kernel)
- Unfused Triton quantize + Triton dequantize (two kernels)

All implementations run on CUDA using Triton kernels.
"""

import gc
import torch

from compressed_tensors.quantization.lifecycle.forward_helpers import (
    _dequantize_triton,
    _quantize_dequantize_triton,
    _quantize_triton,
    adapt_scale_and_zp_for_triton,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils.helpers import calculate_range

SIZE = 4096 * 4096  # ~16.7M elements
device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 200


def create_test_data(rows, cols, quant_type, num_bits, target_device, strategy=QuantizationStrategy.TENSOR, group_size=None):
    """Create test data and quantization parameters."""
    args = QuantizationArgs(
        num_bits=num_bits,
        type=quant_type,
        symmetric=True,
        strategy=strategy,
        group_size=group_size,
    )
    q_min, q_max = calculate_range(args, torch.device(target_device))

    x = torch.randn(rows, cols, dtype=torch.float32, device=target_device)
    
    # Create scale based on strategy
    if strategy == QuantizationStrategy.TENSOR:
        scale = (torch.rand(1) * 0.01 + 0.001).to(target_device)
    elif strategy == QuantizationStrategy.CHANNEL:
        scale = (torch.rand(rows, 1) * 0.01 + 0.001).to(target_device)
    elif strategy == QuantizationStrategy.GROUP:
        num_groups = cols // group_size
        scale = (torch.rand(rows, num_groups) * 0.01 + 0.001).to(target_device)
        x = x.reshape(rows, num_groups, group_size)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    zero_point = None  # symmetric quantization

    return x, scale, zero_point, q_min, q_max, args


def fused_triton_quantize_dequantize(x, scale, zero_point, q_min, q_max, args):
    """Fused Triton quantize+dequantize (single kernel)."""
    num_rows = x.shape[0]
    scale_adapted, zp_adapted = adapt_scale_and_zp_for_triton(scale, zero_point, num_rows)
    return _quantize_dequantize_triton(
        x=x,
        scale=scale_adapted,
        zero_point=zp_adapted,
        q_min=q_min,
        q_max=q_max,
        args=args,
    )


def unfused_triton_quantize_dequantize(x, scale, zero_point, q_min, q_max, args):
    """Unfused Triton: quantize then dequantize (two kernels)."""
    num_rows = x.shape[0]
    scale_adapted, zp_adapted = adapt_scale_and_zp_for_triton(scale, zero_point, num_rows)
    
    # Quantize with Triton
    x_q = _quantize_triton(
        x=x,
        scale=scale_adapted,
        zero_point=zp_adapted,
        q_min=q_min,
        q_max=q_max,
        args=args,
    )
    
    # Dequantize with Triton
    return _dequantize_triton(
        x_q=x_q,
        scale=scale_adapted,
        zero_point=zp_adapted,
        args=args,
    )


def benchmark_cuda(func, x, scale, zero_point, q_min, q_max, args, name, warmup=False):
    """Benchmark a quantize+dequantize function on CUDA using CUDA events for accurate timing."""
    x = x.clone()

    # Warmup phase
    if warmup:
        print(f"  Warming up {name}...")
        for _ in range(50):
            _ = func(x, scale, zero_point, q_min, q_max, args)
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
        result = func(x, scale, zero_point, q_min, q_max, args)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)  # milliseconds
        times.append(elapsed_ms / 1000.0)  # convert to seconds

        del result

    # Use median for robustness against outliers
    times.sort()
    median_time = times[len(times) // 2]
    
    # Print variance info for debugging stability
    min_time = times[0]
    max_time = times[-1]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    variance_ratio = max_time / min_time if min_time > 0 else float('inf')
    print(f"    {name}: median={median_time*1000:.2f}ms, "
          f"min={min_time*1000:.2f}ms, max={max_time*1000:.2f}ms, "
          f"p10={p10*1000:.2f}ms, p90={p90*1000:.2f}ms, "
          f"variance_ratio={variance_ratio:.2f}x")

    return median_time


def run_config(quant_type, num_bits, rows, cols, strategy=QuantizationStrategy.TENSOR, group_size=None):
    """Run benchmarks for a specific configuration."""
    type_str = "int" if quant_type == QuantizationType.INT else "fp"
    
    # Create config name based on strategy
    if strategy == QuantizationStrategy.TENSOR:
        config_name = f"{type_str}{num_bits}"
    elif strategy == QuantizationStrategy.CHANNEL:
        config_name = f"{type_str}{num_bits}_channel"
    elif strategy == QuantizationStrategy.GROUP:
        config_name = f"{type_str}{num_bits}_g{group_size}"
    else:
        config_name = f"{type_str}{num_bits}_{strategy.value}"

    print(f"\n{'='*80}")
    print(f"Benchmarking {config_name} quantize+dequantize ({rows}x{cols} = {rows*cols/1e6:.1f}M elements)")
    print("=" * 80)

    # Create CUDA test data
    x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args = create_test_data(
        rows, cols, quant_type, num_bits, device, strategy, group_size
    )

    # Unfused Triton (two kernels: quantize + dequantize)
    print("\nRunning Unfused Triton (quantize + dequantize)...")
    time_unfused = benchmark_cuda(
        unfused_triton_quantize_dequantize, 
        x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args, 
        "unfused_triton", warmup=True
    )
    print("Unfused Triton:")
    print(f"  Time: {time_unfused*1000:.2f}ms")

    # Fused Triton (single kernel: quantize+dequantize)
    print("\nRunning Fused Triton (single kernel)...")
    time_fused = benchmark_cuda(
        fused_triton_quantize_dequantize,
        x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args,
        "fused_triton", warmup=True
    )
    print("Fused Triton:")
    print(f"  Time: {time_fused*1000:.2f}ms")

    # Verify correctness
    print("\nVerifying correctness...")
    x_test, scale_test, zp_test, q_min_test, q_max_test, args_test = create_test_data(
        512, 1024, quant_type, num_bits, device, strategy, group_size
    )
    
    unfused_out = unfused_triton_quantize_dequantize(
        x_test.clone(), scale_test, zp_test, q_min_test, q_max_test, args_test
    )
    fused_out = fused_triton_quantize_dequantize(
        x_test.clone(), scale_test, zp_test, q_min_test, q_max_test, args_test
    )

    atol = 1e-5
    rtol = 1e-5
    correct = torch.allclose(unfused_out, fused_out, atol=atol, rtol=rtol)
    
    if correct:
        print("  ✓ Results match")
    else:
        diff = (unfused_out - fused_out).abs()
        max_diff = diff.max().item()
        max_idx = diff.argmax()
        print(f"  ✗ Results differ, max_diff={max_diff:.6e}")
        print(f"    unfused={unfused_out.flatten()[max_idx].item():.15f}")
        print(f"    fused={fused_out.flatten()[max_idx].item():.15f}")

    # Calculate speedup
    speedup = time_unfused / time_fused if time_fused > 0 else 0
    print(f"\nSpeedup (unfused/fused): {speedup:.2f}x")

    del x_cuda, scale_cuda, x_test, scale_test
    del unfused_out, fused_out
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "config": config_name,
        "rows": rows,
        "cols": cols,
        "strategy": strategy.value if hasattr(strategy, 'value') else str(strategy),
        "group_size": group_size,
        "unfused_ms": time_unfused * 1000,
        "fused_ms": time_fused * 1000,
        "speedup": speedup,
        "correct": correct,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, Triton requires GPU")
        return

    print("Benchmarking fused vs unfused Triton quantize+dequantize from forward_helpers.py")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"N_RUNS: {N_RUNS}")

    sizes = [
        (4096, 4096),
        (4096, 11008),  # LLaMA MLP
        (8192, 8192),
    ]

    results = []

    # PER-TENSOR (scalar scale)
    print("\n" + "=" * 80)
    print("PER-TENSOR QUANTIZATION (scalar scale)")
    print("=" * 80)
    for quant_type, num_bits in [
        (QuantizationType.INT, 8),
        (QuantizationType.INT, 4),
        (QuantizationType.FLOAT, 4),
        (QuantizationType.FLOAT, 8),
    ]:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols, QuantizationStrategy.TENSOR)
            results.append(result)

    # PER-CHANNEL (one scale per row)
    print("\n" + "=" * 80)
    print("PER-CHANNEL QUANTIZATION (one scale per row)")
    print("=" * 80)
    for quant_type, num_bits in [
        (QuantizationType.INT, 8),
        (QuantizationType.INT, 4),
    ]:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols, QuantizationStrategy.CHANNEL)
            results.append(result)

    # PER-GROUP INT (multiple scales per row)
    print("\n" + "=" * 80)
    print("PER-GROUP INT QUANTIZATION (group_size=128)")
    print("=" * 80)
    for quant_type, num_bits in [
        (QuantizationType.INT, 8),
        (QuantizationType.INT, 4),
    ]:
        for rows, cols in sizes:
            if cols % 128 == 0:  # Only test if divisible by group size
                result = run_config(quant_type, num_bits, rows, cols, QuantizationStrategy.GROUP, group_size=128)
                results.append(result)

    # PER-GROUP FP4 (NVFP4/MXFP4 style)
    print("\n" + "=" * 80)
    print("PER-GROUP FP4 QUANTIZATION (group_size=32)")
    print("=" * 80)
    for rows, cols in sizes:
        if cols % 32 == 0:  # Only test if divisible by group size
            result = run_config(QuantizationType.FLOAT, 4, rows, cols, QuantizationStrategy.GROUP, group_size=32)
            results.append(result)

    # Print summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(
        f"{'Config':<20} {'Size':<15} {'Unfused (ms)':<15} {'Fused (ms)':<15} "
        f"{'Speedup':<10} {'Correct':<10}"
    )
    print("-" * 110)

    for r in results:
        size_str = f"{r['rows']}x{r['cols']}"
        correct_str = "Yes" if r["correct"] else "NO"
        print(
            f"{r['config']:<20} {size_str:<15} {r['unfused_ms']:>12.2f} ms "
            f"{r['fused_ms']:>12.2f} ms "
            f"{r['speedup']:>6.2f}x    {correct_str:<10}"
        )

if __name__ == "__main__":
    main()
