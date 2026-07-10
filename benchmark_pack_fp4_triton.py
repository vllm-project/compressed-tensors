"""
Benchmark script for pack_fp4_to_uint8 implementations.

Compares original (broadcast search) vs current (16-way + int8) vs Triton kernel.
"""

import torch
import time
import gc
import triton
import triton.language as tl

SIZE = 844_000  # ~84.4M elements
device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 200


FLOAT_TO_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@triton.jit
def pack_fp4_kernel_16way(
    x_ptr,
    packed_ptr,
    n_pairs,
    BLOCK_SIZE: tl.constexpr,
):
    # 16-way assignment kernel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pairs

    # Load pairs of values
    low_idx = offsets * 2
    high_idx = offsets * 2 + 1

    x_low = tl.load(x_ptr + low_idx, mask=mask, other=0.0)
    x_high = tl.load(x_ptr + high_idx, mask=mask, other=0.0)

    # Scale and convert
    x_low_scaled = x_low * 2.0
    x_low_int = x_low_scaled.to(tl.int8)

    x_high_scaled = x_high * 2.0
    x_high_int = x_high_scaled.to(tl.int8)

    # Map to indices using cascading tl.where
    # Low nibble
    tmp = tl.full([BLOCK_SIZE], 0, dtype=tl.uint8)
    tmp = tl.where(x_low_int == 1, 1, tmp)
    tmp = tl.where(x_low_int == 2, 2, tmp)
    tmp = tl.where(x_low_int == 3, 3, tmp)
    tmp = tl.where(x_low_int == 4, 4, tmp)
    tmp = tl.where(x_low_int == 6, 5, tmp)
    tmp = tl.where(x_low_int == 8, 6, tmp)
    tmp = tl.where(x_low_int >= 12, 7, tmp)
    tmp = tl.where(x_low_int == -1, 9, tmp)
    tmp = tl.where(x_low_int == -2, 10, tmp)
    tmp = tl.where(x_low_int == -3, 11, tmp)
    tmp = tl.where(x_low_int == -4, 12, tmp)
    tmp = tl.where(x_low_int == -6, 13, tmp)
    tmp = tl.where(x_low_int == -8, 14, tmp)
    idx_low = tl.where(x_low_int <= -12, 15, tmp)

    # High nibble
    tmp = tl.full([BLOCK_SIZE], 0, dtype=tl.uint8)
    tmp = tl.where(x_high_int == 1, 1, tmp)
    tmp = tl.where(x_high_int == 2, 2, tmp)
    tmp = tl.where(x_high_int == 3, 3, tmp)
    tmp = tl.where(x_high_int == 4, 4, tmp)
    tmp = tl.where(x_high_int == 6, 5, tmp)
    tmp = tl.where(x_high_int == 8, 6, tmp)
    tmp = tl.where(x_high_int >= 12, 7, tmp)
    tmp = tl.where(x_high_int == -1, 9, tmp)
    tmp = tl.where(x_high_int == -2, 10, tmp)
    tmp = tl.where(x_high_int == -3, 11, tmp)
    tmp = tl.where(x_high_int == -4, 12, tmp)
    tmp = tl.where(x_high_int == -6, 13, tmp)
    tmp = tl.where(x_high_int == -8, 14, tmp)
    idx_high = tl.where(x_high_int <= -12, 15, tmp)

    # Pack nibbles
    packed = idx_low | (idx_high << 4)

    tl.store(packed_ptr + offsets, packed, mask=mask)


@triton.jit
def pack_fp4_kernel_sign(
    x_ptr,
    packed_ptr,
    n_pairs,
    BLOCK_SIZE: tl.constexpr,
):
    # Sign-based kernel: extract sign, 8-way assignment
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pairs

    # Load pairs of values
    low_idx = offsets * 2
    high_idx = offsets * 2 + 1

    x_low = tl.load(x_ptr + low_idx, mask=mask, other=0.0)
    x_high = tl.load(x_ptr + high_idx, mask=mask, other=0.0)

    # Extract sign (signbit equivalent: x < 0)
    sign_low = (x_low < 0).to(tl.int8)
    sign_high = (x_high < 0).to(tl.int8)

    # Scale and convert to absolute value
    x_low_scaled = x_low * 2.0
    x_low_int = tl.abs(x_low_scaled.to(tl.int8))

    x_high_scaled = x_high * 2.0
    x_high_int = tl.abs(x_high_scaled.to(tl.int8))

    # Map to indices (8-way for positive values only)
    # Low nibble
    tmp = tl.full([BLOCK_SIZE], 0, dtype=tl.uint8)
    tmp = tl.where(x_low_int == 1, 1, tmp)
    tmp = tl.where(x_low_int == 2, 2, tmp)
    tmp = tl.where(x_low_int == 3, 3, tmp)
    tmp = tl.where(x_low_int == 4, 4, tmp)
    tmp = tl.where(x_low_int == 6, 5, tmp)
    tmp = tl.where(x_low_int == 8, 6, tmp)
    idx_low = tl.where(x_low_int >= 12, 7, tmp)
    # Add sign bit
    idx_low = idx_low + (sign_low << 3)

    # High nibble
    tmp = tl.full([BLOCK_SIZE], 0, dtype=tl.uint8)
    tmp = tl.where(x_high_int == 1, 1, tmp)
    tmp = tl.where(x_high_int == 2, 2, tmp)
    tmp = tl.where(x_high_int == 3, 3, tmp)
    tmp = tl.where(x_high_int == 4, 4, tmp)
    tmp = tl.where(x_high_int == 6, 5, tmp)
    tmp = tl.where(x_high_int == 8, 6, tmp)
    idx_high = tl.where(x_high_int >= 12, 7, tmp)
    # Add sign bit
    idx_high = idx_high + (sign_high << 3)

    # Pack nibbles
    packed = idx_low | (idx_high << 4)

    tl.store(packed_ptr + offsets, packed, mask=mask)


@triton.jit
def pack_fp4_kernel_sign_direct(
    x_ptr,
    packed_ptr,
    n_pairs,
    BLOCK_SIZE: tl.constexpr,
):
    # Sign-based kernel with direct computation using threshold counting
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pairs

    # Load pairs
    low_idx = offsets * 2
    high_idx = offsets * 2 + 1

    x_low = tl.load(x_ptr + low_idx, mask=mask, other=0.0)
    x_high = tl.load(x_ptr + high_idx, mask=mask, other=0.0)

    # Extract sign
    sign_low = (x_low < 0).to(tl.uint8)
    sign_high = (x_high < 0).to(tl.uint8)

    # Scale and absolute
    x_low_abs = tl.abs(x_low * 2.0).to(tl.int8)
    x_high_abs = tl.abs(x_high * 2.0).to(tl.int8)

    # Direct index computation via threshold counting
    # Count how many thresholds each value exceeds
    idx_low = (x_low_abs > 1).to(tl.uint8) + (x_low_abs > 2).to(tl.uint8) + \
              (x_low_abs > 4).to(tl.uint8) + (x_low_abs > 6).to(tl.uint8) + \
              (x_low_abs > 10).to(tl.uint8) + (x_low_abs > 13).to(tl.uint8) + \
              (x_low_abs > 20).to(tl.uint8)
    idx_low = idx_low | (sign_low << 3)

    idx_high = (x_high_abs > 1).to(tl.uint8) + (x_high_abs > 2).to(tl.uint8) + \
               (x_high_abs > 4).to(tl.uint8) + (x_high_abs > 6).to(tl.uint8) + \
               (x_high_abs > 10).to(tl.uint8) + (x_high_abs > 13).to(tl.uint8) + \
               (x_high_abs > 20).to(tl.uint8)
    idx_high = idx_high | (sign_high << 3)

    # Pack
    packed = idx_low | (idx_high << 4)

    tl.store(packed_ptr + offsets, packed, mask=mask)


def triton_pack_fp4_to_uint8_16way(x: torch.Tensor) -> torch.Tensor:
    """Triton 16-way kernel."""
    m, n = x.shape
    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    x_flat = x.flatten()
    n_pairs = x_flat.numel() // 2

    packed = torch.empty(n_pairs, dtype=torch.uint8, device=x.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
    pack_fp4_kernel_16way[grid](x_flat, packed, n_pairs, BLOCK_SIZE)

    return packed.reshape(m, n // 2)


def triton_pack_fp4_to_uint8_sign(x: torch.Tensor) -> torch.Tensor:
    """Triton sign-based kernel."""
    m, n = x.shape
    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    x_flat = x.flatten()
    n_pairs = x_flat.numel() // 2

    packed = torch.empty(n_pairs, dtype=torch.uint8, device=x.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
    pack_fp4_kernel_sign[grid](x_flat, packed, n_pairs, BLOCK_SIZE)

    return packed.reshape(m, n // 2)


def triton_pack_fp4_to_uint8_sign_direct(x: torch.Tensor) -> torch.Tensor:
    """Triton sign-based kernel with direct threshold counting."""
    m, n = x.shape
    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    x_flat = x.flatten()
    n_pairs = x_flat.numel() // 2

    packed = torch.empty(n_pairs, dtype=torch.uint8, device=x.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
    pack_fp4_kernel_sign_direct[grid](x_flat, packed, n_pairs, BLOCK_SIZE)

    return packed.reshape(m, n // 2)


def original_pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Original implementation with broadcast search."""
    m, n = x.shape
    device = x.device

    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)

    abs_x = torch.abs(x)
    abs_indices = torch.argmin(torch.abs(abs_x.unsqueeze(-1) - kE2M1), dim=-1).to(torch.int8)
    indices = abs_indices + (torch.signbit(x).to(torch.int8) << 3)
    indices = indices.reshape(-1, 2)
    packed = (indices[:, 0].to(torch.uint8) | (indices[:, 1].to(torch.uint8) << 4))

    return packed.reshape(m, n // 2)


def current_pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Current PyTorch implementation with 16-way assignment + int8."""
    m, n = x.shape

    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    x.mul_(2)
    x = x.to(torch.int8)

    indices = torch.zeros_like(x, dtype=torch.uint8)

    indices[x == 1] = 1
    indices[x == 2] = 2
    indices[x == 3] = 3
    indices[x == 4] = 4
    indices[x == 6] = 5
    indices[x == 8] = 6
    indices[x >= 12] = 7

    indices[x == -1] = 9
    indices[x == -2] = 10
    indices[x == -3] = 11
    indices[x == -4] = 12
    indices[x == -6] = 13
    indices[x == -8] = 14
    indices[x <= -12] = 15

    indices = indices.reshape(-1, 2)
    packed = indices[:, 0] | (indices[:, 1] << 4)

    return packed.reshape(m, n // 2)


def sign_pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Sign-based implementation: extract sign, 8-way assignment + sign bit."""
    m, n = x.shape

    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    

    x.mul_(2)
    x = x.to(torch.int8)

    # Extract sign before conversion
    sign = torch.signbit(x).to(torch.int8)
    x.abs_()

    indices = torch.zeros_like(x, dtype=torch.uint8)

    indices[x == 1] = 1
    indices[x == 2] = 2
    indices[x == 3] = 3
    indices[x == 4] = 4
    indices[x == 6] = 5
    indices[x == 8] = 6
    indices[x >= 12] = 7

    # Apply sign bit (bit 3)
    indices = indices + (sign << 3)

    indices = indices.reshape(-1, 2)
    packed = indices[:, 0] | (indices[:, 1] << 4)

    return packed.reshape(m, n // 2)


def create_test_data(size, device):
    """Create test data with exact FP4 values."""
    x = torch.tensor(FLOAT_TO_E2M1 * (size // 8), dtype=torch.bfloat16, device=device)
    x = x[:size].reshape(-1, 2)
    x[1::2] = -x[1::2]
    return x


def benchmark(func, test_data, name, warmup=False):
    """Benchmark a function."""
    if warmup:
        # Warmup runs to compile kernel
        print(f"  Warming up {name}...")
        warmup_data = create_test_data(1000, device)
        for _ in range(10):
            _ = func(warmup_data.clone())
        del warmup_data
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

        # Record memory before benchmark
        baseline_mem = torch.cuda.memory_allocated(0)

        torch.cuda.synchronize()
        start = time.time()
        result = func(test_data.clone())
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Peak memory excluding baseline
        peak = (torch.cuda.max_memory_allocated(0) - baseline_mem) / 1e9

        times.append(elapsed)
        peaks.append(peak)

        del result
        torch.cuda.empty_cache()
        gc.collect()

    avg_time = sum(times) / N_RUNS
    avg_peak = sum(peaks) / N_RUNS

    return avg_time, avg_peak


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, Triton requires GPU")
        return

    print(f"Benchmarking pack_fp4_to_uint8 implementations ({SIZE/1e6:.1f}M elements)\n")
    print("=" * 80)

    # Create test data once
    print("Creating test data...")
    test_data = create_test_data(SIZE, device)
    print(f"Test data created: {test_data.shape}, {test_data.dtype}\n")

    # Original
    print("Running original implementation...")
    time_orig, peak_orig = benchmark(original_pack_fp4_to_uint8, test_data, "original")
    print(f"Original:")
    print(f"  Time: {time_orig*1000:.2f}ms")
    print(f"  Peak: {peak_orig:.1f} GB")

    # Original compiled
    print(f"\nRunning original (compiled)...")
    compiled_orig = torch.compile(original_pack_fp4_to_uint8)
    time_orig_comp, peak_orig_comp = benchmark(compiled_orig, test_data, "original_compiled", warmup=True)
    print(f"Original (compiled):")
    print(f"  Time: {time_orig_comp*1000:.2f}ms")
    print(f"  Peak: {peak_orig_comp:.1f} GB")

    # Current PyTorch
    print(f"\nRunning current PyTorch implementation...")
    time_curr, peak_curr = benchmark(current_pack_fp4_to_uint8, test_data, "current")
    print(f"Current PyTorch:")
    print(f"  Time: {time_curr*1000:.2f}ms")
    print(f"  Peak: {peak_curr:.1f} GB")

    # Current PyTorch compiled
    print(f"\nRunning current PyTorch (compiled)...")
    compiled_curr = torch.compile(current_pack_fp4_to_uint8)
    time_curr_comp, peak_curr_comp = benchmark(compiled_curr, test_data, "current_compiled", warmup=True)
    print(f"Current PyTorch (compiled):")
    print(f"  Time: {time_curr_comp*1000:.2f}ms")
    print(f"  Peak: {peak_curr_comp:.1f} GB")

    # Sign-based
    print(f"\nRunning sign-based implementation...")
    time_sign, peak_sign = benchmark(sign_pack_fp4_to_uint8, test_data, "sign")
    print(f"Sign-based:")
    print(f"  Time: {time_sign*1000:.2f}ms")
    print(f"  Peak: {peak_sign:.1f} GB")

    # Sign-based compiled
    print(f"\nRunning sign-based (compiled)...")
    compiled_sign = torch.compile(sign_pack_fp4_to_uint8)
    time_sign_comp, peak_sign_comp = benchmark(compiled_sign, test_data, "sign_compiled", warmup=True)
    print(f"Sign-based (compiled):")
    print(f"  Time: {time_sign_comp*1000:.2f}ms")
    print(f"  Peak: {peak_sign_comp:.1f} GB")

    # Triton 16-way
    print(f"\nRunning Triton 16-way kernel...")
    time_triton_16, peak_triton_16 = benchmark(triton_pack_fp4_to_uint8_16way, test_data, "triton_16way", warmup=True)
    print(f"Triton 16-way:")
    print(f"  Time: {time_triton_16*1000:.2f}ms")
    print(f"  Peak: {peak_triton_16:.1f} GB")

    # Triton sign-based
    print(f"\nRunning Triton sign-based kernel...")
    time_triton_sign, peak_triton_sign = benchmark(triton_pack_fp4_to_uint8_sign, test_data, "triton_sign", warmup=True)
    print(f"Triton sign-based:")
    print(f"  Time: {time_triton_sign*1000:.2f}ms")
    print(f"  Peak: {peak_triton_sign:.1f} GB")

    # Triton sign-based direct
    print(f"\nRunning Triton sign-based direct kernel...")
    time_triton_sign_dir, peak_triton_sign_dir = benchmark(triton_pack_fp4_to_uint8_sign_direct, test_data, "triton_sign_direct", warmup=True)
    print(f"Triton sign-based direct:")
    print(f"  Time: {time_triton_sign_dir*1000:.2f}ms")
    print(f"  Peak: {peak_triton_sign_dir:.1f} GB")

    del test_data

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Implementation':<25} {'Time (ms)':<15} {'Peak (GB)':<15} {'Speedup':<15}")
    print("-" * 80)
    print(f"{'Original':<25} {time_orig*1000:>10.2f} ms  {peak_orig:>10.1f} GB   baseline")
    print(f"{'Original (compiled)':<25} {time_orig_comp*1000:>10.2f} ms  {peak_orig_comp:>10.1f} GB   "
          f"{(time_orig/time_orig_comp):>.2f}x")
    print(f"{'Current PyTorch':<25} {time_curr*1000:>10.2f} ms  {peak_curr:>10.1f} GB   "
          f"{(time_orig/time_curr):>.2f}x")
    print(f"{'Current (compiled)':<25} {time_curr_comp*1000:>10.2f} ms  {peak_curr_comp:>10.1f} GB   "
          f"{(time_orig/time_curr_comp):>.2f}x")
    print(f"{'Sign-based':<25} {time_sign*1000:>10.2f} ms  {peak_sign:>10.1f} GB   "
          f"{(time_orig/time_sign):>.2f}x")
    print(f"{'Sign (compiled)':<25} {time_sign_comp*1000:>10.2f} ms  {peak_sign_comp:>10.1f} GB   "
          f"{(time_orig/time_sign_comp):>.2f}x")
    print(f"{'Triton 16-way':<25} {time_triton_16*1000:>10.2f} ms  {peak_triton_16:>10.1f} GB   "
          f"{(time_orig/time_triton_16):>.2f}x")
    print(f"{'Triton sign-based':<25} {time_triton_sign*1000:>10.2f} ms  {peak_triton_sign:>10.1f} GB   "
          f"{(time_orig/time_triton_sign):>.2f}x")
    print(f"{'Triton sign direct':<25} {time_triton_sign_dir*1000:>10.2f} ms  {peak_triton_sign_dir:>10.1f} GB   "
          f"{(time_orig/time_triton_sign_dir):>.2f}x")


if __name__ == "__main__":
    main()


"""

844_000_000
================================================================================
SUMMARY
================================================================================
Implementation            Time (ms)       Peak (GB)       Speedup        
--------------------------------------------------------------------------------
Original                      115.62 ms        37.1 GB   baseline
Original (compiled)            17.75 ms         8.9 GB   6.51x
Current PyTorch                26.38 ms         2.5 GB   4.38x
Current (compiled)              9.21 ms         3.0 GB   12.55x
Sign-based                     17.66 ms         5.1 GB   6.55x
Sign (compiled)                 9.09 ms         3.4 GB   12.72x
Triton 16-way                   2.76 ms         2.1 GB   41.93x
Triton sign-based               2.44 ms         2.1 GB   47.41x
Triton sign direct              2.13 ms         2.1 GB   54.17x

844_000_00
================================================================================
SUMMARY
================================================================================
Implementation            Time (ms)       Peak (GB)       Speedup        
--------------------------------------------------------------------------------
Original                       11.55 ms         3.0 GB   baseline
Original (compiled)            10.13 ms         0.9 GB   1.14x
Current PyTorch                 9.24 ms         0.3 GB   1.25x
Current (compiled)              5.40 ms         0.3 GB   2.14x
Sign-based                      2.16 ms         0.5 GB   5.34x
Sign (compiled)                 5.19 ms         0.3 GB   2.22x
Triton 16-way                   0.76 ms         0.2 GB   15.23x
Triton sign-based               0.77 ms         0.2 GB   14.96x
Triton sign direct              0.72 ms         0.2 GB   16.06x

================================================================================
SUMMARY
================================================================================
Implementation            Time (ms)       Peak (GB)       Speedup        
--------------------------------------------------------------------------------
Original                        1.39 ms         0.0 GB   baseline
Original (compiled)             1.58 ms         0.0 GB   0.88x
Current PyTorch                 1.96 ms         0.0 GB   0.71x
Current (compiled)              1.22 ms         0.0 GB   1.13x
Sign-based                      1.08 ms         0.0 GB   1.29x
Sign (compiled)                 1.14 ms         0.0 GB   1.22x
Triton 16-way                   0.53 ms         0.0 GB   2.59x
Triton sign-based               0.52 ms         0.0 GB   2.69x
Triton sign direct              0.52 ms         0.0 GB   2.69x


844_00
================================================================================
SUMMARY
================================================================================
Implementation            Time (ms)       Peak (GB)       Speedup        
--------------------------------------------------------------------------------
Original                        0.87 ms         0.0 GB   baseline
Original (compiled)             0.96 ms         0.0 GB   0.91x
Current PyTorch                 1.13 ms         0.0 GB   0.77x
Current (compiled)              0.69 ms         0.0 GB   1.27x
Sign-based                      0.35 ms         0.0 GB   2.46x
Sign (compiled)                 0.67 ms         0.0 GB   1.30x
Triton 16-way                   0.15 ms         0.0 GB   5.90x
Triton sign-based               0.16 ms         0.0 GB   5.53x
Triton sign direct              0.14 ms         0.0 GB   6.11x


"""