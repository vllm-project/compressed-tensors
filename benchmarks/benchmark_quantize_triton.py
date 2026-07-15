# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for _quantize Triton implementation in forward_helpers.py.

Compares Triton kernel vs PyTorch ops, both on CUDA (apples to apples).
"""

import gc
import time
import torch

from compressed_tensors.quantization.lifecycle.forward_helpers import _quantize
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils.helpers import calculate_range

SIZE = 4096 * 4096  # ~16.7M elements
device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 200


def create_test_data(rows, cols, quant_type, num_bits, target_device):
    """Create test data and quantization parameters."""
    args = QuantizationArgs(
        num_bits=num_bits,
        type=quant_type,
        symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
    )
    q_min, q_max = calculate_range(args, torch.device(target_device))

    x = torch.randn(rows, cols, dtype=torch.float32, device=target_device)
    scale = (torch.rand(1) * 0.01 + 0.001).to(target_device)
    zero_point = None  # symmetric quantization

    return x, scale, zero_point, q_min, q_max, args


def pytorch_quantize_cuda(x, scale, zero_point, q_min, q_max, args):
    """PyTorch reference implementation on CUDA (no Triton)."""
    from compressed_tensors.quantization.quant_args import round_to_quantized_type_args
    scaled = x / scale
    if zero_point is not None:
        scaled = scaled + zero_point.to(x.dtype)
    return round_to_quantized_type_args(tensor=scaled, args=args, min=q_min, max=q_max)


def benchmark_cuda(func, x, scale, zero_point, q_min, q_max, args, name, warmup=False):
    """Benchmark a quantization function on CUDA."""
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


def run_config(quant_type, num_bits, rows, cols):
    """Run benchmarks for a specific configuration."""
    type_str = "int" if quant_type == QuantizationType.INT else "fp"
    config_name = f"{type_str}{num_bits}"

    print(f"\n{'='*80}")
    print(f"Benchmarking {config_name} quantization ({rows}x{cols} = {rows*cols/1e6:.1f}M elements)")
    print("=" * 80)

    # Create CUDA test data - both paths run on CUDA for fair comparison
    x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args = create_test_data(
        rows, cols, quant_type, num_bits, device
    )

    # PyTorch reference on CUDA (no Triton kernel, just PyTorch ops)
    print("\nRunning PyTorch reference (CUDA, no Triton)...")
    time_pytorch, peak_pytorch = benchmark_cuda(
        pytorch_quantize_cuda, x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args, "pytorch_cuda", warmup=True
    )
    print(f"PyTorch (CUDA):")
    print(f"  Time: {time_pytorch*1000:.2f}ms")
    print(f"  Peak: {peak_pytorch:.3f} GB")

    # Triton kernel (CUDA path in _quantize)
    print("\nRunning Triton kernel (CUDA)...")
    time_triton, peak_triton = benchmark_cuda(
        _quantize, x_cuda, scale_cuda, zp_cuda, q_min_cuda, q_max_cuda, args, "triton", warmup=True
    )
    print(f"Triton (CUDA):")
    print(f"  Time: {time_triton*1000:.2f}ms")
    print(f"  Peak: {peak_triton:.3f} GB")

    # Verify correctness - compare Triton kernel vs PyTorch ops on same CUDA data
    x_test = torch.randn(512, 1024, dtype=torch.float32, device=device)
    scale_test = (torch.rand(1) * 0.01 + 0.001).to(device)

    # PyTorch reference on CUDA
    pytorch_out = pytorch_quantize_cuda(
        x=x_test.clone(),
        scale=scale_test.clone(),
        zero_point=None,
        q_min=q_min_cuda.clone(),
        q_max=q_max_cuda.clone(),
        args=args,
    )

    # Triton kernel on CUDA
    triton_out = _quantize(
        x=x_test.clone(),
        scale=scale_test.clone(),
        zero_point=None,
        q_min=q_min_cuda.clone(),
        q_max=q_max_cuda.clone(),
        args=args,
    )

    # Tolerance depends on quantization type (matching test_quantize_dequantize_matches_sequential)
    if quant_type == QuantizationType.INT:
        atol = 1.0  # Allow 1 unit difference due to rounding at boundaries
    else:
        atol = 0.5  # FP4/FP8 may have boundary differences

    diff = (pytorch_out - triton_out).abs()
    max_diff = diff.max().item()
    correct = max_diff <= atol

    if not correct:
        max_idx = diff.argmax()
        row_idx = max_idx // 1024
        col_idx = max_idx % 1024
        scaled_val = x_test[row_idx, col_idx].item() / scale_test.item()
        print(f"\nWarning: outputs differ, max_diff={max_diff:.6f} (atol={atol})")
        print(f"  At index [{row_idx}, {col_idx}]:")
        print(f"    input={x_test[row_idx, col_idx].item():.15f}")
        print(f"    scale={scale_test.item():.15f}")
        print(f"    scaled={scaled_val:.15f}")
        print(f"    pytorch={pytorch_out[row_idx, col_idx].item():.6f}")
        print(f"    triton={triton_out[row_idx, col_idx].item():.6f}")

    del x_cuda, scale_cuda, q_min_cuda, q_max_cuda
    del x_test, scale_test, pytorch_out, triton_out
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "config": config_name,
        "rows": rows,
        "cols": cols,
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

    print(f"Benchmarking _quantize from forward_helpers.py")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"N_RUNS: {N_RUNS}")

    # Configurations to benchmark
    configs = [
        (QuantizationType.INT, 4),
        (QuantizationType.INT, 8),
        (QuantizationType.FLOAT, 4),
        (QuantizationType.FLOAT, 8),
    ]

    # Tensor sizes
    sizes = [
        (4096, 4096),
        (4096, 11008),  # LLaMA MLP
        (8192, 8192),
    ]

    results = []

    for quant_type, num_bits in configs:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols)
            results.append(result)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY (both on CUDA - apples to apples)")
    print("=" * 100)
    print(f"{'Config':<8} {'Size':<15} {'PyTorch/CUDA (ms)':<18} "
          f"{'Triton/CUDA (ms)':<18} {'Speedup':<10} {'Correct':<8}")
    print("-" * 100)

    for r in results:
        size_str = f"{r['rows']}x{r['cols']}"
        correct_str = "Yes" if r["correct"] else "NO"
        print(f"{r['config']:<8} {size_str:<15} {r['pytorch_ms']:>14.2f} ms  "
              f"{r['triton_ms']:>14.2f} ms  "
              f"{r['speedup']:>6.2f}x    {correct_str:<8}")


if __name__ == "__main__":
    main()


"""
Example output:

====================================================================================================
SUMMARY (both on CUDA - apples to apples)
====================================================================================================
Config   Size            PyTorch/CUDA (ms)  Triton/CUDA (ms)   Speedup    Correct 
----------------------------------------------------------------------------------------------------
int4     4096x4096              X.XX ms           X.XX ms    X.XXx    Yes     
int4     4096x11008             X.XX ms           X.XX ms    X.XXx    Yes     
int8     4096x4096              X.XX ms           X.XX ms    X.XXx    Yes     
fp4      4096x4096              X.XX ms           X.XX ms    X.XXx    Yes     
...
"""
