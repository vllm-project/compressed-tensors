"""Benchmark the fused LUT-B DP with and without final-row endpoint pruning.

The workload models weighted 512-value LUT-B codebook fits. Each tile has its
own values and scale-squared weights. All input generation and fitting run on
the GPU; CUDA events measure end-to-end fitting latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from benchmark_lut_b_triton import dp_divide_conquer_fused_stages_triton


def median_cuda_ms(fn, repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def run_case(tiles: int, n: int, k: int, repeats: int) -> dict:
    values = torch.randn((tiles, n), device="cuda", dtype=torch.float32)
    # Positive, nonuniform scale-squared importance weights, paired with values.
    weights = torch.exp2(
        torch.rand((tiles, n), device="cuda", dtype=torch.float32) * 4.0 - 2.0
    )
    baseline = dp_divide_conquer_fused_stages_triton(
        values, k, weights=weights, candidate_block_size=32
    )
    torch.cuda.synchronize()
    timings = {}
    for block_size in (32, 64, 128, 256, 512):

        def endpoint_fn(block_size=block_size):
            return dp_divide_conquer_fused_stages_triton(
                values,
                k,
                weights=weights,
                candidate_block_size=block_size,
                final_endpoint_only=True,
            )

        centers = endpoint_fn()
        torch.cuda.synchronize()
        max_diff = (centers - baseline).abs().max().item()
        if max_diff != 0.0:
            raise AssertionError(
                f"endpoint-only centers differ at block={block_size}: {max_diff}"
            )
        timings[str(block_size)] = {
            "median_ms": median_cuda_ms(endpoint_fn, repeats),
            "max_center_difference": max_diff,
        }
    baseline_ms = median_cuda_ms(
        lambda: dp_divide_conquer_fused_stages_triton(
            values, k, weights=weights, candidate_block_size=32
        ),
        repeats,
    )
    best_block = min(timings, key=lambda block: timings[block]["median_ms"])
    return {
        "tiles": tiles,
        "points_per_tile": n,
        "centers": k,
        "weighted": True,
        "baseline_full_final_row_ms": baseline_ms,
        "endpoint_only_by_candidate_block": timings,
        "best_endpoint_block": int(best_block),
        "best_endpoint_ms": timings[best_block]["median_ms"],
        "speedup_vs_baseline": baseline_ms / timings[best_block]["median_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=int, nargs="+", default=[512, 4096, 16384])
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/lutb_dp_final_endpoint_benchmark.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    print(f"GPU: {torch.cuda.get_device_name()}; Triton {triton.__version__}")
    results = [run_case(tiles, args.n, args.k, args.repeats) for tiles in args.tiles]
    for result in results:
        print(
            f"tiles={result['tiles']:5d} baseline={result['baseline_full_final_row_ms']:.3f} ms, "
            f"endpoint={result['best_endpoint_ms']:.3f} ms "
            f"(block={result['best_endpoint_block']}, "
            f"{result['speedup_vs_baseline']:.2f}x)"
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
