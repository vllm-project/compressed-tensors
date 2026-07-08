"""
Benchmark script for pack_fp4_to_uint8 implementations.

Compares original (broadcast search) vs current (16-way + int8) implementations
without torch.compile.
"""

import torch
import time
import gc

SIZE = 844_000_000  # Full .844B elements
device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 3


FLOAT_TO_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def original_pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Original implementation with broadcast search."""
    m, n = x.shape
    device = x.device

    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)

    # Original broadcast approach (expensive!)
    abs_x = torch.abs(x)
    abs_indices = torch.argmin(torch.abs(abs_x.unsqueeze(-1) - kE2M1), dim=-1).to(torch.int8)

    # Apply sign bit
    indices = abs_indices + (torch.signbit(x).to(torch.int8) << 3)

    # Reshape and pack
    indices = indices.reshape(-1, 2)
    packed = (indices[:, 0].to(torch.uint8) | (indices[:, 1].to(torch.uint8) << 4))

    return packed.reshape(m, n // 2)


def current_pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Current implementation with 16-way assignment + int8."""
    m, n = x.shape

    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    # Convert to int8 to save memory
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


def create_test_data(size, device):
    """Create test data with exact FP4 values."""
    x = torch.tensor(FLOAT_TO_E2M1 * (size // 8), dtype=torch.bfloat16, device=device)
    x = x[:size].reshape(-1, 2)
    x[1::2] = -x[1::2]  # Add negatives
    return x


def benchmark(func, name):
    """Benchmark a function."""
    times = []
    peaks = []

    for _ in range(N_RUNS):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        x = create_test_data(SIZE, device)

        torch.cuda.synchronize()
        start = time.time()
        result = func(x.clone())
        torch.cuda.synchronize()
        elapsed = time.time() - start

        peak = torch.cuda.max_memory_allocated(0) / 1e9

        times.append(elapsed)
        peaks.append(peak)

        del x, result
        torch.cuda.empty_cache()
        gc.collect()

    avg_time = sum(times) / N_RUNS
    avg_peak = sum(peaks) / N_RUNS

    return avg_time, avg_peak


def main():
    print(f"Benchmarking pack_fp4_to_uint8 implementations ({SIZE/1e9:.2f}B elements)\n")
    print("=" * 80)
    print("NOTE: This benchmark uses full-scale 8.44B elements and will take time.\n")

    # Original
    print("Running original implementation...")
    if torch.cuda.is_available():
        print(f"Memory before original: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    time_orig, peak_orig = benchmark(original_pack_fp4_to_uint8, "original")
    print(f"Original:")
    print(f"  Time: {time_orig:.3f}s")
    print(f"  Peak: {peak_orig:.1f} GB")
    if torch.cuda.is_available():
        print(f"Memory after original: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")

    # Current
    print(f"\nRunning current implementation...")
    if torch.cuda.is_available():
        print(f"Memory before current: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    time_curr, peak_curr = benchmark(current_pack_fp4_to_uint8, "current")
    print(f"Current:")
    print(f"  Time: {time_curr:.3f}s")
    print(f"  Peak: {peak_curr:.1f} GB")
    if torch.cuda.is_available():
        print(f"Memory after current: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Implementation':<20} {'Time (s)':<15} {'Peak (GB)':<15} {'vs Original':<15}")
    print("-" * 80)
    print(f"{'Original':<20} {time_orig:>10.3f} s   {peak_orig:>10.1f} GB   baseline")
    print(f"{'Current':<20} {time_curr:>10.3f} s   {peak_curr:>10.1f} GB   "
          f"{peak_curr - peak_orig:+.1f} GB")

    print(f"\nSpeedup: {time_orig/time_curr:.2f}x")
    print(f"Memory savings: {peak_orig - peak_curr:.1f} GB")


if __name__ == "__main__":
    main()
