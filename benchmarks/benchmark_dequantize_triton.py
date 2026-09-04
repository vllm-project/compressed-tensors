# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for _dequantize Triton implementation in forward_helpers.py.

Compares Triton kernel vs PyTorch ops, both on CUDA (apples to apples).

Based on benchmark_quantize_triton.py structure.
"""

import gc
import torch

from compressed_tensors.quantization.lifecycle.forward_helpers import _dequantize
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils.helpers import calculate_range

device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 200


def create_test_data(
    rows,
    cols,
    quant_type,
    num_bits,
    target_device,
    strategy=QuantizationStrategy.TENSOR,
    group_size=None,
):
    """Create quantized test data and dequantization parameters."""
    args = QuantizationArgs(
        num_bits=num_bits,
        type=quant_type,
        symmetric=True,
        strategy=strategy,
        group_size=group_size,
    )
    q_min, q_max = calculate_range(args, torch.device(target_device))

    # Create quantized values within the valid range
    x_q = torch.randint(
        int(q_min.item()),
        int(q_max.item()) + 1,
        (rows, cols),
        dtype=torch.float32,
        device=target_device,
    )

    # Create scale based on strategy
    if strategy == QuantizationStrategy.TENSOR:
        scale = (torch.rand(1) * 0.01 + 0.001).to(target_device)
        zero_point = None
    elif strategy == QuantizationStrategy.CHANNEL:
        scale = (torch.rand(rows, 1) * 0.01 + 0.001).to(target_device)
        zero_point = None
    elif strategy == QuantizationStrategy.GROUP:
        num_groups = cols // group_size
        scale = (torch.rand(rows, num_groups) * 0.01 + 0.001).to(target_device)
        zero_point = None
        x_q = x_q.reshape(rows, num_groups, group_size)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return x_q, scale, zero_point, args


def pytorch_dequantize_cuda(x_q, scale, zero_point, args):
    """PyTorch reference implementation on CUDA (no Triton)."""
    scale_broadcast = scale
    zp_broadcast = zero_point

    # Ensure scale broadcasts correctly to x_q shape
    while scale_broadcast.ndim < x_q.ndim:
        scale_broadcast = scale_broadcast.unsqueeze(-1)
    if zp_broadcast is not None:
        while zp_broadcast.ndim < x_q.ndim:
            zp_broadcast = zp_broadcast.unsqueeze(-1)

    dequant_value = x_q.to(scale_broadcast.dtype)
    if zp_broadcast is not None:
        dequant_value = dequant_value - zp_broadcast.to(scale_broadcast.dtype)
    dequant_value = dequant_value * scale_broadcast

    return dequant_value


def triton_dequantize_cuda(x_q, scale, zero_point, args):
    """Triton kernel wrapper."""
    return _dequantize(
        x_q=x_q,
        scale=scale,
        zero_point=zero_point,
        args=args,
    )


def benchmark_cuda(func, x_q, scale, zero_point, args, name, warmup=False):
    """Benchmark a dequantization function on CUDA using CUDA events for accurate timing."""
    x_q = x_q.clone()

    # Warmup phase
    if warmup:
        print(f"  Warming up {name}...")
        for _ in range(50):
            _ = func(x_q, scale, zero_point, args)
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
        result = func(x_q, scale, zero_point, args)
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
    variance_ratio = max_time / min_time if min_time > 0 else float("inf")
    print(
        f"    {name}: median={median_time*1000:.2f}ms, "
        f"min={min_time*1000:.2f}ms, max={max_time*1000:.2f}ms, "
        f"p10={p10*1000:.2f}ms, p90={p90*1000:.2f}ms, "
        f"variance_ratio={variance_ratio:.2f}x"
    )

    return median_time


def run_config(
    quant_type,
    num_bits,
    rows,
    cols,
    strategy=QuantizationStrategy.TENSOR,
    group_size=None,
):
    """Run benchmarks for a specific configuration."""
    type_str = "int" if quant_type == QuantizationType.INT else "fp"
    strategy_str = (
        strategy.value if hasattr(strategy, "value") else str(strategy).split(".")[-1].lower()
    )

    if strategy == QuantizationStrategy.GROUP:
        config_name = f"{type_str}{num_bits}_g{group_size}"
    else:
        config_name = f"{type_str}{num_bits}_{strategy_str}"

    print(f"\n{'='*80}")
    print(f"Benchmarking {config_name} dequantization ({rows}x{cols} = {rows*cols/1e6:.1f}M elements)")
    print("=" * 80)

    # Create CUDA test data
    x_q_cuda, scale_cuda, zp_cuda, args = create_test_data(
        rows, cols, quant_type, num_bits, device, strategy, group_size
    )

    # PyTorch reference on CUDA (no Triton kernel, just PyTorch ops)
    print("\nRunning PyTorch reference (CUDA, no Triton)...")
    time_pytorch = benchmark_cuda(
        pytorch_dequantize_cuda, x_q_cuda, scale_cuda, zp_cuda, args, "pytorch_cuda", warmup=True
    )
    print("PyTorch (CUDA):")
    print(f"  Time: {time_pytorch*1000:.2f}ms")

    # Triton kernel (CUDA path in _dequantize)
    print("\nRunning Triton kernel (CUDA)...")
    time_triton = benchmark_cuda(
        triton_dequantize_cuda, x_q_cuda, scale_cuda, zp_cuda, args, "triton", warmup=True
    )
    print("Triton (CUDA):")
    print(f"  Time: {time_triton*1000:.2f}ms")

    # Verify correctness
    test_rows, test_cols = 512, 1024
    x_q_test, scale_test, zp_test, args_test = create_test_data(
        test_rows, test_cols, quant_type, num_bits, device, strategy, group_size
    )

    pytorch_out = pytorch_dequantize_cuda(x_q_test.clone(), scale_test.clone(), zp_test, args_test)
    triton_out = triton_dequantize_cuda(x_q_test.clone(), scale_test.clone(), zp_test, args_test)

    atol = 1e-5
    rtol = 1e-5

    diff = (pytorch_out - triton_out).abs()
    max_diff = diff.max().item()
    correct = torch.allclose(pytorch_out, triton_out, atol=atol, rtol=rtol)

    if not correct:
        max_idx = diff.argmax()
        print(f"\nWarning: outputs differ, max_diff={max_diff:.6f} (atol={atol})")
        print(f"  pytorch={pytorch_out.flatten()[max_idx].item():.15f}")
        print(f"  triton={triton_out.flatten()[max_idx].item():.15f}")

    del x_q_cuda, scale_cuda, x_q_test, scale_test, pytorch_out, triton_out
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
        "speedup": time_pytorch / time_triton if time_triton > 0 else 0,
        "correct": correct,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, Triton requires GPU")
        return

    print("Benchmarking _dequantize from forward_helpers.py")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"N_RUNS: {N_RUNS}")

    sizes = [
        (4096, 4096),
        (4096, 11008),  # LLaMA MLP
        (8192, 8192),
    ]

    results = []

    # Per-tensor (scalar scale) - uses fast scalar kernel path
    print("\n" + "=" * 80)
    print("PER-TENSOR (scalar scale) - uses fast scalar kernel path")
    print("=" * 80)
    for quant_type, num_bits in [(QuantizationType.INT, 8), (QuantizationType.INT, 4)]:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols, QuantizationStrategy.TENSOR)
            results.append(result)

    # Per-channel - uses strided kernel path
    print("\n" + "=" * 80)
    print("PER-CHANNEL (one scale per row) - uses strided kernel path")
    print("=" * 80)
    for quant_type, num_bits in [(QuantizationType.INT, 8), (QuantizationType.INT, 4)]:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols, QuantizationStrategy.CHANNEL)
            results.append(result)

    # Per-group - uses strided kernel path
    print("\n" + "=" * 80)
    print("PER-GROUP (multiple scales per row) - uses strided kernel path")
    print("=" * 80)
    for quant_type, num_bits, group_size in [
        (QuantizationType.INT, 8, 128),
        (QuantizationType.INT, 4, 128),
    ]:
        for rows, cols in sizes:
            result = run_config(
                quant_type, num_bits, rows, cols, QuantizationStrategy.GROUP, group_size
            )
            results.append(result)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':<15} {'Size':<15} {'PyTorch (ms)':<15} "
        f"{'Triton (ms)':<15} {'Speedup':<10} {'Correct':<8}"
    )
    print("-" * 100)

    for r in results:
        size_str = f"{r['rows']}x{r['cols']}"
        correct_str = "Yes" if r["correct"] else "NO"
        print(
            f"{r['config']:<15} {size_str:<15} {r['pytorch_ms']:>11.2f} ms  "
            f"{r['triton_ms']:>11.2f} ms  "
            f"{r['speedup']:>6.2f}x    {correct_str:<8}"
        )


if __name__ == "__main__":
    main()
