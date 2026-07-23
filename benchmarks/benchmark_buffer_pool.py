"""
Benchmark script for pack_fp4_to_uint8 buffer pool comparison.

Tests the Triton kernel with vs without buffer pool across realistic LLM tensor sizes.
"""

import torch
import gc

from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    QuantBufferPool,
)

# Realistic tensor sizes based on LLM architectures
TEST_SIZES = {
    "Small (1M)":        1_000_000,        # ~1M elements - baseline
    "7B Attention":     16_777_216,        # 4096 × 4096 = 16.7M
    "7B MLP":           45_088_768,        # 11008 × 4096 = 45M  
    "70B Attention":    67_108_864,        # 8192 × 8192 = 67M
    "70B MLP":         234_881_024,        # 28672 × 8192 = 235M
    "405B Attention":  268_435_456,        # 16384 × 16384 = 268M
    "405B MLP":        872_415_232,        # 53248 × 16384 = 872M
}

FLOAT_TO_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

device = "cuda:0" if torch.accelerator.is_available() else "cpu"


def create_test_data(size, device):
    """Create test data with exact FP4 values."""
    x = torch.tensor(FLOAT_TO_E2M1 * ((size + 7) // 8), dtype=torch.bfloat16, device=device)
    x = x[:size].reshape(-1, 2)
    x[1::2] = -x[1::2]
    return x


def triton_with_buffer_pool(x: torch.Tensor) -> torch.Tensor:
    """Triton kernel using buffer pool (avoids cudaMalloc)."""
    return pack_fp4_to_uint8(x, use_buffer_pool=True)


def triton_without_buffer_pool(x: torch.Tensor) -> torch.Tensor:
    """Triton kernel without buffer pool (fresh allocation each call)."""
    return pack_fp4_to_uint8(x, use_buffer_pool=False)


def benchmark_size(size, name, n_runs=None):
    """Benchmark buffer pool vs no buffer pool for a given size."""
    if n_runs is None:
        # Adjust runs based on size (fewer for large tensors)
        if size > 200_000_000:
            n_runs = 20
        elif size > 50_000_000:
            n_runs = 30
        else:
            n_runs = 50
    
    print(f"\n{'='*80}")
    print(f"Testing: {name} ({size/1e6:.1f}M elements, ~{size*2/1e6:.1f}MB input, ~{size/2/1e6:.1f}MB output)")
    print(f"{'='*80}")
    
    # Create test data
    test_data = create_test_data(size, device)
    
    # Warmup both kernels with full-size tensor
    # Using the actual test size ensures Triton kernel is warmed up for the
    # correct grid configuration and memory access patterns.
    print("  Warming up kernels...")
    for _ in range(5):
        _ = triton_with_buffer_pool(test_data)
        _ = triton_without_buffer_pool(test_data)
    torch.cuda.synchronize()
    
    # Clear buffer pool to start fresh
    QuantBufferPool.clear()
    
    # Interleaved benchmark using CUDA events for accurate GPU timing.
    # Alternating between pool and no-pool eliminates ordering bias from
    # GPU warmup, thermal throttling, and memory state differences.
    # Note: We don't call torch.cuda.empty_cache() because we want to measure
    # the realistic comparison: QuantBufferPool vs PyTorch's native caching
    # allocator. Clearing the cache would force expensive cudaMalloc calls.
    times_pool = []
    times_no_pool = []
    for _ in range(n_runs):
        # Run with buffer pool
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = triton_with_buffer_pool(test_data)
        end_event.record()
        torch.cuda.synchronize()
        times_pool.append(start_event.elapsed_time(end_event) / 1000.0)  # ms to seconds
        del result
        
        # Run without buffer pool
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = triton_without_buffer_pool(test_data)
        end_event.record()
        torch.cuda.synchronize()
        times_no_pool.append(start_event.elapsed_time(end_event) / 1000.0)  # ms to seconds
        del result
    
    del test_data
    torch.cuda.empty_cache()
    gc.collect()
    
    avg_pool = sum(times_pool) / len(times_pool)
    avg_no_pool = sum(times_no_pool) / len(times_no_pool)
    
    return {
        "name": name,
        "size": size,
        "time_pool": avg_pool,
        "time_no_pool": avg_no_pool,
    }


def main():
    if not torch.accelerator.is_available():
        print("CUDA not available, Triton requires GPU")
        return

    print("="*80)
    print("BUFFER POOL BENCHMARK - Realistic LLM Tensor Sizes")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Testing pack_fp4_to_uint8 with vs without buffer pool")
    
    results = []
    
    for name, size in TEST_SIZES.items():
        try:
            result = benchmark_size(size, name)
            results.append(result)
            
            speedup = result["time_no_pool"] / result["time_pool"] if result["time_pool"] > 0 else 0
            diff_ms = (result["time_no_pool"] - result["time_pool"]) * 1000
            
            print(f"\n  With buffer pool:    {result['time_pool']*1000:>8.3f} ms")
            print(f"  Without buffer pool: {result['time_no_pool']*1000:>8.3f} ms")
            print(f"  Buffer pool effect:  {speedup:.3f}x ({diff_ms:+.3f} ms)")
            
        except torch.cuda.OutOfMemoryError:
            print(f"  SKIPPED - Out of memory")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Final summary table
    print("\n")
    print("="*100)
    print("SUMMARY: Buffer Pool Effect Across Tensor Sizes")
    print("="*100)
    print(f"{'Tensor Size':<20} {'Elements':<12} {'Output':<10} {'With Pool':<12} {'No Pool':<12} {'Effect':<10} {'Diff':<10}")
    print("-"*100)
    
    for r in results:
        speedup = r["time_no_pool"] / r["time_pool"] if r["time_pool"] > 0 else 0
        diff_ms = (r["time_no_pool"] - r["time_pool"]) * 1000
        output_mb = r["size"] / 2 / 1e6
        
        effect_str = f"{speedup:.2f}x" if speedup >= 1 else f"{speedup:.2f}x"
        diff_str = f"{diff_ms:+.2f}ms"
        
        print(f"{r['name']:<20} {r['size']/1e6:>8.1f}M   {output_mb:>6.1f}MB  "
              f"{r['time_pool']*1000:>8.2f}ms   {r['time_no_pool']*1000:>8.2f}ms   "
              f"{effect_str:>8}   {diff_str:>8}")
    
    print("-"*100)
    print("\nNote: Effect > 1.0x means buffer pool is FASTER, < 1.0x means SLOWER")
    print("      Positive diff means buffer pool saves time, negative means overhead")


if __name__ == "__main__":
    main()
