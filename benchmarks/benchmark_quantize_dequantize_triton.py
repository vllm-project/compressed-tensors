# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for fused _quantize_dequantize Triton implementation.

Compares Triton kernel vs PyTorch ops, both on CUDA (apples to apples).
This is the key operation used in fake_quantize() during QAT/calibration.
"""

import gc
import time
import torch

from compressed_tensors.quantization.lifecycle.forward_helpers import (
    _quantize_dequantize,
    round_to_quantized_type_args,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils.helpers import calculate_range

device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 200


def create_test_data(
    rows, cols, quant_type, num_bits, target_device,
    strategy=QuantizationStrategy.TENSOR, group_size=None
):
    """Create test data and quantization parameters."""
    args = QuantizationArgs(
        num_bits=num_bits,
        type=quant_type,
        symmetric=True,
        strategy=strategy,
        group_size=group_size,
    )
    q_min, q_max = calculate_range(args, torch.device(target_device))

    # Create input tensor (random float values to quantize)
    x = torch.randn(rows, cols, dtype=torch.float32, device=target_device)

    # Create scale based on strategy
    if strategy == QuantizationStrategy.TENSOR:
        scale = (torch.rand(1) * 0.1 + 0.01).to(target_device)
        zero_point = None
    elif strategy == QuantizationStrategy.CHANNEL:
        scale = (torch.rand(rows, 1) * 0.1 + 0.01).to(target_device)
        zero_point = None
    elif strategy == QuantizationStrategy.GROUP:
        num_groups = cols // group_size
        scale = (torch.rand(rows, num_groups) * 0.1 + 0.01).to(target_device)
        zero_point = None
        # Reshape to 3D as _process_group would do
        x = x.reshape(rows, num_groups, group_size)
        scale = scale.unsqueeze(-1)  # (rows, num_groups, 1)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return x, scale, zero_point, q_min, q_max, args, group_size


def pytorch_quantize_dequantize_cuda(x, scale, zero_point, q_min, q_max, args, global_scale=None):
    """
    PyTorch reference implementation on CUDA (no Triton).
    Mirrors the CPU fallback path in _quantize_dequantize.
    """
    effective_scale = scale
    if global_scale is not None:
        effective_scale = scale / global_scale

    scaled = x / effective_scale

    if zero_point is not None:
        scaled = scaled + zero_point.to(x.dtype)

    # clamp and round
    quantized = round_to_quantized_type_args(
        tensor=scaled, args=args, min=q_min, max=q_max
    )

    # dequantize
    dequant = quantized.to(effective_scale.dtype)
    if zero_point is not None:
        dequant = dequant - zero_point.to(effective_scale.dtype)

    return dequant * effective_scale


def benchmark_cuda(func, x, scale, zero_point, q_min, q_max, args, name, warmup=False):
    """Benchmark a quantize-dequantize function on CUDA."""
    if warmup:
        print(f"  Warming up {name}...")
        for _ in range(10):
            _ = func(x, scale, zero_point, q_min, q_max, args)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print(f"  Warmup complete, starting benchmark...")

    times = []
    peaks = []

    for _ in range(N_RUNS):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        baseline_mem = torch.cuda.memory_allocated(0)

        torch.cuda.synchronize()
        start = time.time()
        result = func(x, scale, zero_point, q_min, q_max, args)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        peak = (torch.cuda.max_memory_allocated(0) - baseline_mem) / 1e9

        times.append(elapsed)
        peaks.append(peak)

        del result
        torch.cuda.empty_cache()
        gc.collect()

    avg_time = sum(times) / N_RUNS
    avg_peak = sum(peaks) / N_RUNS

    return avg_time, avg_peak


def triton_quantize_dequantize(x, scale, zero_point, q_min, q_max, args):
    """Wrapper for _quantize_dequantize (dispatches to Triton on CUDA)."""
    return _quantize_dequantize(
        x=x,
        scale=scale,
        zero_point=zero_point,
        q_min=q_min,
        q_max=q_max,
        args=args,
        global_scale=None,
    )


def run_config(quant_type, num_bits, rows, cols, strategy=QuantizationStrategy.TENSOR, group_size=None):
    """Run benchmarks for a specific configuration."""
    type_str = "int" if quant_type == QuantizationType.INT else "fp"
    strategy_str = strategy.value if hasattr(strategy, 'value') else str(strategy).split('.')[-1].lower()

    if strategy == QuantizationStrategy.GROUP:
        config_name = f"{type_str}{num_bits}_g{group_size}"
    else:
        config_name = f"{type_str}{num_bits}_{strategy_str}"

    # FP8 falls back to CPU even on CUDA (no Triton kernel support)
    is_fp8 = quant_type == QuantizationType.FLOAT and num_bits == 8
    if is_fp8:
        print(f"\nSkipping {config_name} - FP8 requires CPU fallback (no Triton support)")
        return None

    print(f"\n{'='*80}")
    print(f"Benchmarking {config_name} quantize_dequantize ({rows}x{cols} = {rows*cols/1e6:.1f}M elements)")
    print("=" * 80)

    # Create CUDA test data
    x_cuda, scale_cuda, zp_cuda, q_min, q_max, args, gs = create_test_data(
        rows, cols, quant_type, num_bits, device, strategy, group_size
    )

    # PyTorch reference on CUDA (no Triton kernel, just PyTorch ops)
    print("\nRunning PyTorch reference (CUDA, no Triton)...")
    time_pytorch, peak_pytorch = benchmark_cuda(
        pytorch_quantize_dequantize_cuda,
        x_cuda.clone(), scale_cuda.clone(), zp_cuda,
        q_min, q_max, args, "pytorch_cuda", warmup=True
    )
    print(f"PyTorch (CUDA):")
    print(f"  Time: {time_pytorch*1000:.2f}ms")
    print(f"  Peak: {peak_pytorch:.3f} GB")

    # Triton kernel (CUDA path in _quantize_dequantize)
    print("\nRunning Triton kernel (CUDA)...")
    time_triton, peak_triton = benchmark_cuda(
        triton_quantize_dequantize,
        x_cuda.clone(), scale_cuda.clone(), zp_cuda,
        q_min, q_max, args, "triton", warmup=True
    )
    print(f"Triton (CUDA):")
    print(f"  Time: {time_triton*1000:.2f}ms")
    print(f"  Peak: {peak_triton:.3f} GB")

    # Verify correctness
    test_rows, test_cols = 512, 1024
    x_test, scale_test, zp_test, q_min_test, q_max_test, args_test, _ = create_test_data(
        test_rows, test_cols, quant_type, num_bits, device, strategy, group_size
    )

    pytorch_out = pytorch_quantize_dequantize_cuda(
        x_test.clone(), scale_test.clone(), zp_test, q_min_test, q_max_test, args_test
    )
    triton_out = triton_quantize_dequantize(
        x_test.clone(), scale_test.clone(), zp_test, q_min_test, q_max_test, args_test
    )

    atol = 1e-5
    rtol = 1e-5
    diff = (pytorch_out - triton_out).abs()
    max_diff = diff.max().item()
    correct = torch.allclose(pytorch_out, triton_out, atol=atol, rtol=rtol)

    if not correct:
        print(f"\nWarning: outputs differ, max_diff={max_diff:.6f} (atol={atol})")

    del x_cuda, scale_cuda, x_test, scale_test, pytorch_out, triton_out
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "config": config_name,
        "rows": rows,
        "cols": cols,
        "strategy": strategy_str,
        "group_size": group_size,
        "pytorch_ms": time_pytorch * 1000,
        "triton_ms": time_triton * 1000,
        "triton_peak": peak_triton,
        "speedup": time_pytorch / time_triton if time_triton > 0 else 0,
        "correct": correct,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, Triton requires GPU")
        return

    print(f"Benchmarking fused _quantize_dequantize (fake_quantize path)")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"N_RUNS: {N_RUNS}")

    # Tensor sizes (cols must be divisible by group sizes)
    sizes = [
        (4096, 4096),
        (4096, 11008),  # LLaMA MLP
        (8192, 8192),
    ]

    results = []

    # Per-tensor (scalar scale) - uses fast scalar kernel
    print("\n" + "=" * 80)
    print("PER-TENSOR (scalar scale) - uses fast scalar kernel path")
    print("=" * 80)
    tensor_configs = [
        (QuantizationType.INT, 4),
        (QuantizationType.INT, 8),
        (QuantizationType.FLOAT, 4),
        (QuantizationType.FLOAT, 8),
    ]
    for quant_type, num_bits in tensor_configs:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols, QuantizationStrategy.TENSOR)
            if result is not None:
                results.append(result)

    # Per-channel - uses grouped kernel
    print("\n" + "=" * 80)
    print("PER-CHANNEL (one scale per row) - uses grouped kernel path")
    print("=" * 80)
    channel_configs = [
        (QuantizationType.INT, 8),
        (QuantizationType.INT, 4),
    ]
    for quant_type, num_bits in channel_configs:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols, QuantizationStrategy.CHANNEL)
            if result is not None:
                results.append(result)

    # Per-group - uses grouped kernel
    print("\n" + "=" * 80)
    print("PER-GROUP (multiple scales per row) - uses grouped kernel path")
    print("=" * 80)
    group_configs = [
        (QuantizationType.INT, 8, 128),
        (QuantizationType.INT, 4, 128),
        (QuantizationType.INT, 4, 64),
    ]
    for quant_type, num_bits, group_size in group_configs:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols, QuantizationStrategy.GROUP, group_size)
            if result is not None:
                results.append(result)

    # Print summary
    print("\n" + "=" * 110)
    print("SUMMARY (both on CUDA - apples to apples)")
    print("=" * 110)
    print(f"{'Config':<15} {'Size':<15} {'PyTorch/CUDA (ms)':<18} "
          f"{'Triton/CUDA (ms)':<18} {'Speedup':<10} {'Correct':<8}")
    print("-" * 110)

    for r in results:
        size_str = f"{r['rows']}x{r['cols']}"
        correct_str = "Yes" if r["correct"] else "NO"
        print(f"{r['config']:<15} {size_str:<15} {r['pytorch_ms']:>14.2f} ms  "
              f"{r['triton_ms']:>14.2f} ms  "
              f"{r['speedup']:>6.2f}x    {correct_str:<8}")


if __name__ == "__main__":
    main()
