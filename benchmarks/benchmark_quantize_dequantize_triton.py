# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for fused _quantize_dequantize Triton implementation.

Compares:
- Fused Triton quantize+dequantize (single kernel)
- Unfused Triton quantize + Triton dequantize (two kernels)
- Mixed Triton quantize + PyTorch dequantize
- Pure PyTorch quantize + PyTorch dequantize

All implementations run on CUDA.
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
        x,
        scale_adapted,
        zero_point=zp_adapted,
        q_min=q_min,
        q_max=q_max,
        args=args,
    )


def pytorch_quantize_cuda(x, scale, zero_point, q_min, q_max, args):
    """PyTorch quantization on CUDA (no Triton)."""
    # Ensure scale broadcasts correctly to x shape
    scale_broadcast = scale
    while scale_broadcast.ndim < x.ndim:
        scale_broadcast = scale_broadcast.unsqueeze(-1)
    
    zp_broadcast = zero_point
    if zp_broadcast is not None:
        while zp_broadcast.ndim < x.ndim:
            zp_broadcast = zp_broadcast.unsqueeze(-1)
    
    # Quantize: round(x / scale + zero_point)
    if zp_broadcast is not None:
        quant_value = x / scale_broadcast + zp_broadcast.to(x.dtype)
    else:
        quant_value = x / scale_broadcast
    
    # Handle different quantization types
    from compressed_tensors.quantization.quant_args import QuantizationType
    if args.type == QuantizationType.FLOAT:
        # Float quantization (FP4 or FP8)
        quant_value = torch.clamp(quant_value, q_min, q_max)
        if args.num_bits == 4:
            # FP4 E2M1: Map to nearest representable value
            from compressed_tensors.quantization.utils.fp4_utils import cast_to_fp4
            quant_value = cast_to_fp4(quant_value)
        elif args.num_bits == 8:
            # FP8: Cast to float8 dtype then back
            quant_value = quant_value.to(torch.float8_e4m3fn).to(x.dtype)
    else:
        # Integer quantization
        quant_value = torch.round(quant_value)
        quant_value = torch.clamp(quant_value, q_min, q_max)
    
    return quant_value


def pytorch_dequantize_cuda(x_q, scale, zero_point, args):
    """PyTorch dequantization on CUDA (no Triton)."""
    # Ensure scale broadcasts correctly to x_q shape
    scale_broadcast = scale
    while scale_broadcast.ndim < x_q.ndim:
        scale_broadcast = scale_broadcast.unsqueeze(-1)
    
    zp_broadcast = zero_point
    if zp_broadcast is not None:
        while zp_broadcast.ndim < x_q.ndim:
            zp_broadcast = zp_broadcast.unsqueeze(-1)
    
    # Dequantize: (x_q - zero_point) * scale
    dequant_value = x_q.to(scale_broadcast.dtype)
    if zp_broadcast is not None:
        dequant_value = dequant_value - zp_broadcast.to(scale_broadcast.dtype)
    dequant_value = dequant_value * scale_broadcast
    
    return dequant_value


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


def mixed_triton_pytorch_quantize_dequantize(x, scale, zero_point, q_min, q_max, args):
    """Mixed: Triton quantize + PyTorch dequantize."""
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
    
    # Dequantize with PyTorch
    return pytorch_dequantize_cuda(x_q, scale, zero_point, args)


def pytorch_quantize_dequantize(x, scale, zero_point, q_min, q_max, args):
    """Pure PyTorch: quantize then dequantize (no Triton)."""
    # Quantize with PyTorch
    x_q = pytorch_quantize_cuda(x, scale, zero_point, q_min, q_max, args)
    
    # Dequantize with PyTorch
    return pytorch_dequantize_cuda(x_q, scale, zero_point, args)


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

    # 1. Pure PyTorch (both quantize + dequantize)
    print("\nRunning PyTorch quantize + PyTorch dequantize...")
    time_pytorch = benchmark_cuda(
        pytorch_quantize_dequantize,
        x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args,
        "pytorch_q+d", warmup=True
    )
    print("PyTorch Q+D:")
    print(f"  Time: {time_pytorch*1000:.2f}ms")

    # 2. Mixed Triton quantize + PyTorch dequantize
    print("\nRunning Triton quantize + PyTorch dequantize...")
    time_mixed = benchmark_cuda(
        mixed_triton_pytorch_quantize_dequantize,
        x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args,
        "triton_q+pytorch_d", warmup=True
    )
    print("Triton Q + PyTorch D:")
    print(f"  Time: {time_mixed*1000:.2f}ms")

    # 3. Unfused Triton (both quantize + dequantize)
    print("\nRunning Triton quantize + Triton dequantize...")
    time_unfused = benchmark_cuda(
        unfused_triton_quantize_dequantize, 
        x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args, 
        "triton_q+d", warmup=True
    )
    print("Triton Q+D (unfused):")
    print(f"  Time: {time_unfused*1000:.2f}ms")

    # 4. Fused Triton (single kernel: quantize+dequantize)
    print("\nRunning Fused Triton (single kernel)...")
    time_fused = benchmark_cuda(
        fused_triton_quantize_dequantize,
        x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args,
        "triton_fused", warmup=True
    )
    print("Triton Fused:")
    print(f"  Time: {time_fused*1000:.2f}ms")

    # Verify correctness
    print("\nVerifying correctness...")
    x_test, scale_test, zp_test, q_min_test, q_max_test, args_test = create_test_data(
        512, 1024, quant_type, num_bits, device, strategy, group_size
    )
    
    pytorch_out = pytorch_quantize_dequantize(
        x_test.clone(), scale_test, zp_test, q_min_test, q_max_test, args_test
    )
    mixed_out = mixed_triton_pytorch_quantize_dequantize(
        x_test.clone(), scale_test, zp_test, q_min_test, q_max_test, args_test
    )
    unfused_out = unfused_triton_quantize_dequantize(
        x_test.clone(), scale_test, zp_test, q_min_test, q_max_test, args_test
    )
    fused_out = fused_triton_quantize_dequantize(
        x_test.clone(), scale_test, zp_test, q_min_test, q_max_test, args_test
    )

    atol = 1e-5
    rtol = 1e-5
    
    # Check all against PyTorch reference
    mixed_correct = torch.allclose(mixed_out, pytorch_out, atol=atol, rtol=rtol)
    unfused_correct = torch.allclose(unfused_out, pytorch_out, atol=atol, rtol=rtol)
    fused_correct = torch.allclose(fused_out, pytorch_out, atol=atol, rtol=rtol)
    
    all_correct = mixed_correct and unfused_correct and fused_correct
    
    if all_correct:
        print("  ✓ All results match PyTorch reference")
    else:
        print("  ✗ Some results differ:")
        if not mixed_correct:
            diff = (mixed_out - pytorch_out).abs()
            print(f"    Mixed vs PyTorch max_diff={diff.max().item():.6e}")
        if not unfused_correct:
            diff = (unfused_out - pytorch_out).abs()
            print(f"    Unfused vs PyTorch max_diff={diff.max().item():.6e}")
        if not fused_correct:
            diff = (fused_out - pytorch_out).abs()
            print(f"    Fused vs PyTorch max_diff={diff.max().item():.6e}")

    # Calculate speedups vs PyTorch baseline
    speedup_mixed = time_pytorch / time_mixed if time_mixed > 0 else 0
    speedup_unfused = time_pytorch / time_unfused if time_unfused > 0 else 0
    speedup_fused = time_pytorch / time_fused if time_fused > 0 else 0
    
    print(f"\nSpeedup vs PyTorch baseline:")
    print(f"  Mixed (Triton Q + PyTorch D): {speedup_mixed:.2f}x")
    print(f"  Triton Unfused: {speedup_unfused:.2f}x")
    print(f"  Triton Fused: {speedup_fused:.2f}x")

    del x_cuda, scale_cuda, x_test, scale_test
    del pytorch_out, mixed_out, unfused_out, fused_out
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "config": config_name,
        "rows": rows,
        "cols": cols,
        "strategy": strategy.value if hasattr(strategy, 'value') else str(strategy),
        "group_size": group_size,
        "pytorch_ms": time_pytorch * 1000,
        "mixed_ms": time_mixed * 1000,
        "unfused_ms": time_unfused * 1000,
        "fused_ms": time_fused * 1000,
        "speedup_mixed": speedup_mixed,
        "speedup_unfused": speedup_unfused,
        "speedup_fused": speedup_fused,
        "correct": all_correct,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, Triton requires GPU")
        return

    from compressed_tensors.utils.triton import HAS_TRITON

    if not HAS_TRITON:
        print("Triton is not available, skipping benchmark")
        return

    print("Benchmarking quantize+dequantize implementations:")
    print("  1. PyTorch Q + PyTorch D (baseline)")
    print("  2. Triton Q + PyTorch D (mixed)")
    print("  3. Triton Q + Triton D (unfused)")
    print("  4. Triton Fused Q+D (single kernel)")
    print(f"\nDevice: {torch.cuda.get_device_name(device)}")
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
    print("\n" + "=" * 145)
    print("SUMMARY - All times in ms, speedups relative to PyTorch baseline")
    print("=" * 145)
    print(
        f"{'Config':<20} {'Size':<12} {'PyTorch':<10} {'Mixed':<10} {'T Unfused':<10} "
        f"{'T Fused':<10} {'Mix SpUp':<9} {'Unf SpUp':<9} {'Fused SpUp':<11} {'OK':<4}"
    )
    print("-" * 145)

    for r in results:
        size_str = f"{r['rows']}x{r['cols']}"
        correct_str = "Yes" if r["correct"] else "NO"
        print(
            f"{r['config']:<20} {size_str:<12} "
            f"{r['pytorch_ms']:>7.2f} ms {r['mixed_ms']:>7.2f} ms "
            f"{r['unfused_ms']:>7.2f} ms {r['fused_ms']:>7.2f} ms "
            f"{r['speedup_mixed']:>6.2f}x  {r['speedup_unfused']:>6.2f}x  "
            f"{r['speedup_fused']:>6.2f}x      {correct_str:<4}"
        )

    # Print mode descriptions
    print("\n" + "=" * 145)
    print("BENCHMARKED MODES")
    print("=" * 145)
    print()
    print("  PyTorch:    PyTorch quantize + PyTorch dequantize (native PyTorch ops, baseline)")
    print("  Mixed:      Triton quantize + PyTorch dequantize")
    print("  T Unfused:  Triton quantize + Triton dequantize (two separate kernels)")
    print("  T Fused:    Triton quantize+dequantize (single fused kernel)")
    print()

if __name__ == "__main__":
    main()
