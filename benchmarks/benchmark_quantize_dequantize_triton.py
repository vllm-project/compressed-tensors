# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for fused _quantize_dequantize Triton implementation.

Compares:
- Fused Triton quantize+dequantize (single kernel)
- Triton quantize followed by Triton dequantize (two kernels)
- Triton quantize followed by PyTorch dequantize

All implementations run on CUDA for apples-to-apples comparison.
"""

import gc
import time
import torch

from compressed_tensors.quantization.lifecycle.forward_helpers import (
    _dequantize,
    _quantize,
    _quantize_dequantize,
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
    rows,
    cols,
    quant_type,
    num_bits,
    target_device,
    strategy=QuantizationStrategy.TENSOR,
    group_size=None,
    block_structure=None,
    symmetric=True,
):
    """Create test data for quantize+dequantize benchmarks."""
    args = QuantizationArgs(
        num_bits=num_bits,
        type=quant_type,
        symmetric=symmetric,
        strategy=strategy,
        group_size=group_size,
        block_structure=block_structure,
    )
    q_min, q_max = calculate_range(args, torch.device(target_device))

    # Create random float input data
    x = torch.randn(rows, cols, dtype=torch.float32, device=target_device)

    # Create scale and zero_point based on strategy
    if strategy == QuantizationStrategy.TENSOR:
        scale = (torch.rand(1) * 0.01 + 0.001).to(target_device)
        if symmetric:
            zero_point = None
        else:
            zero_point = torch.zeros(1, device=target_device)
    elif strategy == QuantizationStrategy.CHANNEL:
        scale = (torch.rand(rows, 1) * 0.01 + 0.001).to(target_device)
        if symmetric:
            zero_point = None
        else:
            zero_point = torch.zeros(rows, 1, device=target_device)
    elif strategy == QuantizationStrategy.GROUP:
        num_groups = cols // group_size
        scale = (torch.rand(rows, num_groups) * 0.01 + 0.001).to(target_device)
        if symmetric:
            zero_point = None
        else:
            zero_point = torch.zeros(rows, num_groups, device=target_device)
        x = x.reshape(rows, num_groups, group_size)
    elif strategy == QuantizationStrategy.BLOCK:
        block_rows, block_cols = block_structure
        n_row_blocks = rows // block_rows
        n_col_blocks = cols // block_cols
        scale = (torch.rand(n_row_blocks, n_col_blocks) * 0.01 + 0.001).to(target_device)
        if symmetric:
            zero_point = None
        else:
            zero_point = torch.zeros(n_row_blocks, n_col_blocks, device=target_device)
        x = x.reshape(n_row_blocks, n_col_blocks, block_rows, block_cols)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return x, scale, zero_point, q_min, q_max, args


def fused_triton_quantize_dequantize(x, scale, zero_point, q_min, q_max, args):
    """Fused Triton quantize+dequantize (single kernel pass)."""
    return _quantize_dequantize(
        x=x,
        scale=scale,
        zero_point=zero_point,
        q_min=q_min,
        q_max=q_max,
        args=args,
    )


def triton_quantize_triton_dequantize(x, scale, zero_point, q_min, q_max, args):
    """Triton quantize followed by Triton dequantize (two kernel passes)."""
    x_q = _quantize(
        x=x,
        scale=scale,
        zero_point=zero_point,
        q_min=q_min,
        q_max=q_max,
        args=args,
    )
    return _dequantize(
        x_q=x_q,
        scale=scale,
        zero_point=zero_point,
        args=args,
    )


def pytorch_dequantize_cuda(x_q, scale, zero_point, dtype=None, global_scale=None):
    """PyTorch dequantize implementation on CUDA (no Triton).

    Matches the non-Triton fallback path in forward_helpers._dequantize.
    """
    if global_scale is not None:
        scale = scale / global_scale

    dequant_value = x_q.to(scale.dtype)

    # Ensure scale broadcasts correctly to x_q shape
    while scale.ndim < dequant_value.ndim:
        scale = scale.unsqueeze(-1)
    if zero_point is not None:
        while zero_point.ndim < dequant_value.ndim:
            zero_point = zero_point.unsqueeze(-1)

    if zero_point is not None:
        dequant_value = dequant_value - zero_point.to(scale.dtype)

    dequant_value = dequant_value * scale

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value


def triton_quantize_pytorch_dequantize(x, scale, zero_point, q_min, q_max, args):
    """Triton quantize followed by PyTorch dequantize."""
    x_q = _quantize(
        x=x,
        scale=scale,
        zero_point=zero_point,
        q_min=q_min,
        q_max=q_max,
        args=args,
    )
    return pytorch_dequantize_cuda(x_q, scale, zero_point)


def benchmark_cuda(func, x, scale, zero_point, q_min, q_max, args, name, warmup=False):
    """Benchmark a quantize+dequantize function on CUDA."""
    x = x.clone()
    if warmup:
        print(f"  Warming up {name}...")
        for _ in range(10):
            _ = func(x, scale, zero_point, q_min, q_max, args)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print(f"  Warmup complete, starting benchmark...")

    times = []

    for _ in range(N_RUNS):
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.synchronize()
        start = time.time()
        result = func(x, scale, zero_point, q_min, q_max, args)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        times.append(elapsed)

        del result
        torch.cuda.empty_cache()
        gc.collect()

    avg_time = sum(times) / N_RUNS
    return avg_time


def run_config(
    quant_type,
    num_bits,
    rows,
    cols,
    strategy=QuantizationStrategy.TENSOR,
    group_size=None,
    block_structure=None,
    symmetric=True,
):
    """Run benchmarks for a specific configuration."""
    type_str = "int" if quant_type == QuantizationType.INT else "fp"
    sym_str = "" if symmetric else "_asym"
    strategy_str = (
        strategy.value if hasattr(strategy, "value") else str(strategy).split(".")[-1].lower()
    )

    if strategy == QuantizationStrategy.GROUP:
        config_name = f"{type_str}{num_bits}_g{group_size}{sym_str}"
    elif strategy == QuantizationStrategy.BLOCK:
        config_name = f"{type_str}{num_bits}_b{block_structure[0]}x{block_structure[1]}{sym_str}"
    else:
        config_name = f"{type_str}{num_bits}_{strategy_str}{sym_str}"

    print(f"\n{'='*80}")
    print(f"Benchmarking {config_name} quantize+dequantize ({rows}x{cols} = {rows*cols/1e6:.1f}M elements)")
    print("=" * 80)

    # Create CUDA test data
    x_cuda, scale_cuda, zp_cuda, q_min, q_max, args = create_test_data(
        rows, cols, quant_type, num_bits, device, strategy, group_size, block_structure, symmetric
    )

    # Fused Triton quantize+dequantize
    print("\nRunning fused Triton quantize+dequantize...")
    time_fused = benchmark_cuda(
        fused_triton_quantize_dequantize,
        x_cuda, scale_cuda, zp_cuda, q_min, q_max, args,
        "fused_triton", warmup=True
    )
    print(f"Fused Triton: {time_fused*1000:.2f}ms")

    # Triton quantize + Triton dequantize
    print("\nRunning Triton quantize + Triton dequantize...")
    time_triton_triton = benchmark_cuda(
        triton_quantize_triton_dequantize,
        x_cuda, scale_cuda, zp_cuda, q_min, q_max, args,
        "triton_triton", warmup=True
    )
    print(f"Triton + Triton: {time_triton_triton*1000:.2f}ms")

    # Triton quantize + PyTorch dequantize
    print("\nRunning Triton quantize + PyTorch dequantize...")
    time_triton_pytorch = benchmark_cuda(
        triton_quantize_pytorch_dequantize,
        x_cuda, scale_cuda, zp_cuda, q_min, q_max, args,
        "triton_pytorch", warmup=True
    )
    print(f"Triton + PyTorch: {time_triton_pytorch*1000:.2f}ms")

    # Verify correctness - all three should produce the same output
    # Use smaller test size for BLOCK strategy to avoid shape issues
    if strategy == QuantizationStrategy.BLOCK:
        test_rows = block_structure[0] * 4
        test_cols = block_structure[1] * 8
    else:
        test_rows, test_cols = 512, 1024
    x_test, scale_test, zp_test, q_min_test, q_max_test, args_test = create_test_data(
        test_rows, test_cols, quant_type, num_bits, device, strategy, group_size,
        block_structure, symmetric
    )

    fused_out = fused_triton_quantize_dequantize(
        x_test.clone(), scale_test.clone(), zp_test, q_min_test, q_max_test, args_test
    )
    triton_triton_out = triton_quantize_triton_dequantize(
        x_test.clone(), scale_test.clone(), zp_test, q_min_test, q_max_test, args_test
    )
    triton_pytorch_out = triton_quantize_pytorch_dequantize(
        x_test.clone(), scale_test.clone(), zp_test, q_min_test, q_max_test, args_test
    )

    atol = 1e-5
    rtol = 1e-5

    # Compare fused vs triton+triton
    diff_tt = (fused_out - triton_triton_out).abs()
    correct_tt = torch.allclose(fused_out, triton_triton_out, atol=atol, rtol=rtol)
    if not correct_tt:
        max_diff = diff_tt.max().item()
        max_idx = diff_tt.argmax()
        print(f"\nWarning: fused vs triton+triton differ, max_diff={max_diff:.6f}")
        print(f"  fused={fused_out.flatten()[max_idx].item():.15f}")
        print(f"  triton+triton={triton_triton_out.flatten()[max_idx].item():.15f}")

    # Compare fused vs triton+pytorch
    diff_tp = (fused_out - triton_pytorch_out).abs()
    correct_tp = torch.allclose(fused_out, triton_pytorch_out, atol=atol, rtol=rtol)
    if not correct_tp:
        max_diff = diff_tp.max().item()
        max_idx = diff_tp.argmax()
        print(f"\nWarning: fused vs triton+pytorch differ, max_diff={max_diff:.6f}")
        print(f"  fused={fused_out.flatten()[max_idx].item():.15f}")
        print(f"  triton+pytorch={triton_pytorch_out.flatten()[max_idx].item():.15f}")

    del x_cuda, scale_cuda, x_test, scale_test
    del fused_out, triton_triton_out, triton_pytorch_out
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "config": config_name,
        "rows": rows,
        "cols": cols,
        "strategy": strategy_str,
        "group_size": group_size,
        "fused_ms": time_fused * 1000,
        "triton_triton_ms": time_triton_triton * 1000,
        "triton_pytorch_ms": time_triton_pytorch * 1000,
        "speedup_vs_tt": time_triton_triton / time_fused if time_fused > 0 else 0,
        "speedup_vs_tp": time_triton_pytorch / time_fused if time_fused > 0 else 0,
        "correct_tt": correct_tt,
        "correct_tp": correct_tp,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, Triton requires GPU")
        return

    print("Benchmarking fused _quantize_dequantize from forward_helpers.py")
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

    # Per-group INT - uses strided kernel path
    print("\n" + "=" * 80)
    print("PER-GROUP INT (multiple scales per row) - uses strided kernel path")
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

    # Per-group FP4 - NVFP4/MXFP4 style quantization
    print("\n" + "=" * 80)
    print("PER-GROUP FP4 (NVFP4/MXFP4 style) - uses strided kernel path")
    print("=" * 80)
    for group_size in [16, 32]:
        for rows, cols in sizes:
            result = run_config(
                QuantizationType.FLOAT, 4, rows, cols, QuantizationStrategy.GROUP, group_size
            )
            results.append(result)

    # Asymmetric INT4 with zero_point - W4A16_ASYM style
    print("\n" + "=" * 80)
    print("ASYMMETRIC INT4 (W4A16_ASYM style) - with zero_point")
    print("=" * 80)
    for rows, cols in sizes:
        result = run_config(
            QuantizationType.INT, 4, rows, cols, QuantizationStrategy.GROUP, 128,
            symmetric=False
        )
        results.append(result)

    # Block strategy - DeepSeek-style (128x128 blocks)
    # Note: FP8 falls back to PyTorch (not Triton) due to hardware dtype requirements
    # So we test INT8 BLOCK which uses the Triton kernel
    print("\n" + "=" * 80)
    print("BLOCK INT8 (DeepSeek-style 128x128 blocks) - uses Triton kernel")
    print("=" * 80)
    block_sizes = [
        (4096, 4096),
        (8192, 8192),
    ]
    for rows, cols in block_sizes:
        # Skip if dimensions don't divide evenly by block size
        if rows % 128 == 0 and cols % 128 == 0:
            result = run_config(
                QuantizationType.INT, 8, rows, cols, QuantizationStrategy.BLOCK,
                block_structure=[128, 128]
            )
            results.append(result)

    # Print summary
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)
    print(
        f"{'Config':<20} {'Size':<15} {'Fused (ms)':<12} "
        f"{'T+T (ms)':<12} {'T+Py (ms)':<12} "
        f"{'vs T+T':<10} {'vs T+Py':<10} {'Correct':<10}"
    )
    print("-" * 130)

    for r in results:
        size_str = f"{r['rows']}x{r['cols']}"
        correct_str = "Yes" if r["correct_tt"] and r["correct_tp"] else "NO"
        print(
            f"{r['config']:<20} {size_str:<15} {r['fused_ms']:>9.2f} ms "
            f"{r['triton_triton_ms']:>9.2f} ms "
            f"{r['triton_pytorch_ms']:>9.2f} ms "
            f"{r['speedup_vs_tt']:>6.2f}x   "
            f"{r['speedup_vs_tp']:>6.2f}x   {correct_str:<10}"
        )


if __name__ == "__main__":
    main()
