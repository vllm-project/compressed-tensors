"""Benchmark LUT-B fitters on one real Qwen3-30B-A3B weight matrix.

All tile extraction, normalization, fitting, packing, and error calculations run
on CUDA. The Hugging Face safetensors reader loads the selected source tensor on
CPU before it is copied to the benchmark GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from benchmark_lut_b_triton import (
    dp_divide_conquer_triton,
    dp_divide_conquer_fused_stages_triton,
    dp_divide_conquer_parallel_scan_triton,
    dp_smawk_triton,
    dp_triton,
    emeans_triton,
    lloyd_triton,
    make_emeans_permutations,
)
from compressed_tensors.quantization.quant_scheme import LUTB
from compressed_tensors.quantization.utils.lut_b import (
    _assign_indices,
    _fit_codebooks,
    _scale_lut_b_tiles,
    _weight_to_tiles,
    pack_lut_b_indices,
    quantize_lut_b,
)


def benchmark_gpu(fn, repeats: int) -> float:
    """Return median CUDA event latency in milliseconds."""
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


def load_model_tensor(repo_id: str, tensor_key: str) -> tuple[torch.Tensor, str]:
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as index_file:
        weight_map = json.load(index_file)["weight_map"]
    if tensor_key not in weight_map:
        examples = [
            key for key in weight_map if key.endswith("self_attn.q_proj.weight")
        ]
        raise KeyError(
            f"{tensor_key!r} is absent from {repo_id}; attention q_proj examples: "
            f"{examples[:3]}"
        )
    shard = weight_map[tensor_key]
    shard_path = hf_hub_download(repo_id, shard)
    with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
        tensor = checkpoint.get_tensor(tensor_key)
    return tensor, shard


def weight_space_mse(
    normalized_tiles: torch.Tensor,
    centers: torch.Tensor,
    scale_bytes: torch.Tensor,
) -> torch.Tensor:
    """Nearest-codebook reconstruction MSE in the original weight units."""
    errors = torch.full_like(normalized_tiles, float("inf"))
    for center in range(centers.shape[1]):
        delta = normalized_tiles - centers[:, center, None]
        errors = torch.minimum(errors, delta.square())
    scale = torch.exp2(scale_bytes.to(torch.float32) - 127.0)
    return (errors * scale[:, None].square()).mean()


def importance_weighted_mse(
    normalized_tiles: torch.Tensor,
    centers: torch.Tensor,
    scale_bytes: torch.Tensor,
    importance_tiles: torch.Tensor,
) -> torch.Tensor:
    """Importance-weighted reconstruction MSE in original weight units."""
    errors = torch.full_like(normalized_tiles, float("inf"))
    for center in range(centers.shape[1]):
        delta = normalized_tiles - centers[:, center, None]
        errors = torch.minimum(errors, delta.square())
    scale = torch.exp2(scale_bytes.to(torch.float32) - 127.0)
    weighted_scale = importance_tiles * scale[:, None].square()
    return (errors * weighted_scale).sum() / importance_tiles.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument(
        "--tensor-key", default="model.layers.0.self_attn.q_proj.weight"
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--emeans-epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--candidate-block-sizes",
        type=int,
        nargs="+",
        choices=(16, 32, 64, 128),
        default=(16, 32, 64, 128),
        help="parallel candidate scan widths to benchmark",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="also benchmark weighted 1D k-means using a layer-derived column proxy",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    device = torch.device("cuda")
    source_weight, shard = load_model_tensor(args.repo, args.tensor_key)
    original_shape = tuple(source_weight.shape)
    weight = source_weight.to(device=device, non_blocking=True)
    del source_weight

    quant_args = LUTB["weights"]
    raw_tiles = _weight_to_tiles(weight, quant_args).to(torch.float32)
    normalized_tiles, scale_bytes = _scale_lut_b_tiles(raw_tiles)
    tiles, tile_size = normalized_tiles.shape
    codebook_size = 1 << quant_args.num_bits
    signal_mse = weight.to(torch.float32).square().mean()
    torch.cuda.synchronize()

    # Run CT's production CUDA path once for the reference and once as a
    # benchmarked end-to-end quantization path (including normalization,
    # codebook fitting, index assignment, and packing).
    ct_packed, ct_codebooks, ct_scales = quantize_lut_b(
        weight, quant_args
    )
    if not torch.equal(ct_scales.reshape(-1), scale_bytes):
        raise RuntimeError("reference scaling disagrees with CT's LUT-B scales")
    ct_centers = ct_codebooks.reshape(-1, codebook_size).to(torch.float32)

    def ct_fit_chunked():
        fitted = torch.empty(
            (tiles, codebook_size), device=device, dtype=torch.float32
        )
        for start in range(0, tiles, 4096):
            end = min(start + 4096, tiles)
            _, centers = _fit_codebooks(normalized_tiles[start:end], quant_args)
            fitted[start:end] = centers.to(torch.float32)
        return fitted

    def triton_full_quantize(fitter):
        layer_tiles = _weight_to_tiles(weight, quant_args).to(torch.float32)
        normalized, _ = _scale_lut_b_tiles(layer_tiles)
        centers = fitter(normalized)
        if isinstance(centers, tuple):
            centers = centers[0]
        indices = _assign_indices(normalized, centers)
        return pack_lut_b_indices(indices.to(torch.uint8), quant_args)

    lloyd_centers = lloyd_triton(normalized_tiles, codebook_size, args.iterations)
    dp_centers = dp_triton(normalized_tiles, codebook_size)
    dc_centers = dp_divide_conquer_triton(normalized_tiles, codebook_size)
    dc_parallel_centers = dp_divide_conquer_parallel_scan_triton(
        normalized_tiles, codebook_size
    )
    dc_fused_centers = dp_divide_conquer_fused_stages_triton(
        normalized_tiles, codebook_size
    )
    smawk_centers = dp_smawk_triton(normalized_tiles, codebook_size)
    emeans_centers, recipe = emeans_triton(
        normalized_tiles,
        codebook_size,
        args.emeans_epochs,
        seed=2026,
        batch_law="throughput",
    )

    # A separate measurement excludes random permutation generation, which is
    # setup work for the e-means fitting loop. Also measure it included.
    permutations = make_emeans_permutations(
        tiles,
        tile_size,
        args.emeans_epochs,
        device,
        seed=2026,
    )
    torch.cuda.synchronize()
    emeans_triton(
        normalized_tiles,
        codebook_size,
        args.emeans_epochs,
        permutations=permutations,
        batch_law="throughput",
    )

    torch.cuda.synchronize()
    candidates = {
        "CT DP (production)": ct_centers,
        "Triton Lloyd": lloyd_centers,
        "Triton DP k-means": dp_centers,
        "Triton DP divide-and-conquer": dc_centers,
        "Triton DP D&C parallel scan": dc_parallel_centers,
        "Triton DP D&C fused stages": dc_fused_centers,
        "Triton DP SMAWK": smawk_centers,
        "Triton e-means": emeans_centers,
    }
    mse = {
        name: weight_space_mse(normalized_tiles, centers, scale_bytes).item()
        for name, centers in candidates.items()
    }
    torch.cuda.synchronize()

    # Compiling and warming each path happens before timing. All calls process
    # the full layer tile batch in parallel; there is no per-tile host loop.
    fit_timings = {
        "CT DP fit (4096 chunks)": benchmark_gpu(
            ct_fit_chunked, args.repeats
        ),
        "Triton Lloyd fit": benchmark_gpu(
            lambda: lloyd_triton(normalized_tiles, codebook_size, args.iterations),
            args.repeats,
        ),
        "DP exhaustive O(kn^2) fit": benchmark_gpu(
            lambda: dp_triton(normalized_tiles, codebook_size), args.repeats
        ),
        "DP divide-conquer O(knlogn) fit": benchmark_gpu(
            lambda: dp_divide_conquer_triton(normalized_tiles, codebook_size),
            args.repeats,
        ),
        "DP D&C parallel fused stages fit": benchmark_gpu(
            lambda: dp_divide_conquer_fused_stages_triton(
                normalized_tiles, codebook_size
            ),
            args.repeats,
        ),
        "DP SMAWK O(kn) fit": benchmark_gpu(
            lambda: dp_smawk_triton(normalized_tiles, codebook_size), args.repeats
        ),
        "Triton e-means fit": benchmark_gpu(
            lambda: emeans_triton(
                normalized_tiles,
                codebook_size,
                args.emeans_epochs,
                permutations=permutations,
                batch_law="throughput",
            ),
            args.repeats,
        ),
        "Triton e-means incl shuffle": benchmark_gpu(
            lambda: emeans_triton(
                normalized_tiles,
                codebook_size,
                args.emeans_epochs,
                seed=2026,
                batch_law="throughput",
            ),
            args.repeats,
        ),
    }
    for block_size in args.candidate_block_sizes:
        fit_timings[f"DP D&C scan width {block_size}"] = benchmark_gpu(
            lambda block_size=block_size: dp_divide_conquer_parallel_scan_triton(
                normalized_tiles,
                codebook_size,
                candidate_block_size=block_size,
            ),
            args.repeats,
        )

    full_timings = {
        "CT DP production": benchmark_gpu(
            lambda: quantize_lut_b(weight, quant_args),
            args.repeats,
        ),
        "Triton Lloyd full quantize": benchmark_gpu(
            lambda: triton_full_quantize(
                lambda values: lloyd_triton(
                    values, codebook_size, args.iterations
                )
            ),
            args.repeats,
        ),
        "DP exhaustive O(kn^2) full": benchmark_gpu(
            lambda: triton_full_quantize(
                lambda values: dp_triton(values, codebook_size)
            ),
            args.repeats,
        ),
        "DP divide-conquer O(knlogn) full": benchmark_gpu(
            lambda: triton_full_quantize(
                lambda values: dp_divide_conquer_triton(values, codebook_size)
            ),
            args.repeats,
        ),
        "DP D&C parallel scan full": benchmark_gpu(
            lambda: triton_full_quantize(
                lambda values: dp_divide_conquer_parallel_scan_triton(
                    values, codebook_size
                )
            ),
            args.repeats,
        ),
        "DP D&C fused stages full": benchmark_gpu(
            lambda: triton_full_quantize(
                lambda values: dp_divide_conquer_fused_stages_triton(
                    values, codebook_size
                )
            ),
            args.repeats,
        ),
        "DP SMAWK O(kn) full": benchmark_gpu(
            lambda: triton_full_quantize(
                lambda values: dp_smawk_triton(values, codebook_size)
            ),
            args.repeats,
        ),
        "Triton e-means full quantize": benchmark_gpu(
            lambda: triton_full_quantize(
                lambda values: emeans_triton(
                    values,
                    codebook_size,
                    args.emeans_epochs,
                    seed=2026,
                    batch_law="throughput",
                )
            ),
            args.repeats,
        ),
    }

    print(f"GPU: {torch.cuda.get_device_name()} | Triton {triton.__version__}")
    print(f"Model: {args.repo} | tensor: {args.tensor_key} | shard: {shard}")
    print(
        f"Weight: {original_shape} {weight.dtype}; LUT-B tiles: {tiles:,} x "
        f"{tile_size} (8x64); {weight.numel():,} values"
    )
    print(
        f"e-means: throughput batch law, batch={recipe['batch_size']}, "
        f"epochs={args.emeans_epochs}, alpha0={recipe['alpha0']:.4f}"
    )
    print("Timings are median CUDA event milliseconds; lower is faster.")
    print("Fit-only timings (normalized tiles are precomputed):")
    print(f"{'Method':32s} {'ms':>10s} {'Mweights/s':>13s}")
    for name, latency in fit_timings.items():
        throughput = weight.numel() / (latency * 1000.0)
        print(f"{name:32s} {latency:10.3f} {throughput:13.1f}")

    print("Full layer quantization timings (tile, scale, fit, assign, pack):")
    print(
        f"{'Method':32s} {'ms':>10s} {'Mweights/s':>13s} "
        f"{'MSE':>14s} {'MSE/signal':>13s}"
    )
    for name, latency in full_timings.items():
        quality_name = {
            "CT DP production": "CT DP (production)",
            "Triton Lloyd full quantize": "Triton Lloyd",
            "DP exhaustive O(kn^2) full": "Triton DP k-means",
            "DP divide-conquer O(knlogn) full": "Triton DP divide-and-conquer",
            "DP D&C parallel scan full": "Triton DP D&C parallel scan",
            "DP D&C fused stages full": "Triton DP D&C fused stages",
            "DP SMAWK O(kn) full": "Triton DP SMAWK",
            "Triton e-means full quantize": "Triton e-means",
        }[name]
        throughput = weight.numel() / (latency * 1000.0)
        print(
            f"{name:32s} {latency:10.3f} {throughput:13.1f} "
            f"{mse[quality_name]:14.6g} "
            f"{mse[quality_name] / signal_mse.item() * 100:12.3f}%"
        )
    print(f"Packed CT output: {ct_packed.numel():,} bytes")

    if args.weighted:
        rows, columns = original_shape
        row_tiles, column_tiles = rows // 8, columns // 64
        # No calibration activations are loaded here. Use the layer's per-input
        # channel RMS as a positive, nonuniform importance proxy.
        column_importance = weight.to(torch.float32).square().mean(dim=0).sqrt()
        column_importance = (
            column_importance / column_importance.mean()
        ).clamp_min(0.05)
        column_importance = column_importance / column_importance.mean()
        importance_tiles = (
            column_importance.reshape(column_tiles, 64)
            .unsqueeze(0)
            .expand(row_tiles, -1, -1)
            .unsqueeze(2)
            .expand(-1, -1, 8, -1)
            .reshape(tiles, tile_size)
            .contiguous()
        )

        weighted_ct_packed, weighted_ct_codebooks, weighted_ct_scales = (
            quantize_lut_b(
                weight,
                quant_args,
                importance=column_importance,
            )
        )
        if not torch.equal(weighted_ct_scales.reshape(-1), scale_bytes):
            raise RuntimeError("weighted CT quantization changed LUT-B scales")
        weighted_ct_centers = weighted_ct_codebooks.reshape(
            -1, codebook_size
        ).to(torch.float32)
        weighted_candidates = {
            "CT weighted DP": weighted_ct_centers,
            "DP exhaustive O(kn^2)": dp_triton(
                normalized_tiles, codebook_size, weights=importance_tiles
            ),
            "DP divide-conquer O(knlogn)": dp_divide_conquer_triton(
                normalized_tiles, codebook_size, weights=importance_tiles
            ),
            "DP D&C parallel scan": dp_divide_conquer_parallel_scan_triton(
                normalized_tiles, codebook_size, weights=importance_tiles
            ),
            "DP D&C fused stages": dp_divide_conquer_fused_stages_triton(
                normalized_tiles, codebook_size, weights=importance_tiles
            ),
            "DP SMAWK O(kn)": dp_smawk_triton(
                normalized_tiles, codebook_size, weights=importance_tiles
            ),
        }
        weighted_mse = {
            name: importance_weighted_mse(
                normalized_tiles, centers, scale_bytes, importance_tiles
            ).item()
            for name, centers in weighted_candidates.items()
        }
        weighted_signal = (
            (raw_tiles.square() * importance_tiles).sum()
            / importance_tiles.sum()
        ).item()

        def weighted_triton_full(fitter):
            layer_tiles = _weight_to_tiles(weight, quant_args).to(torch.float32)
            normalized, _ = _scale_lut_b_tiles(layer_tiles)
            centers = fitter(normalized, weights=importance_tiles)
            indices = _assign_indices(normalized, centers)
            return pack_lut_b_indices(indices.to(torch.uint8), quant_args)

        weighted_full_timings = {
            "CT weighted DP": benchmark_gpu(
                lambda: quantize_lut_b(
                    weight,
                    quant_args,
                    importance=column_importance,
                ),
                args.repeats,
            ),
            "DP exhaustive O(kn^2)": benchmark_gpu(
                lambda: weighted_triton_full(
                    lambda values, weights: dp_triton(
                        values, codebook_size, weights=weights
                    )
                ),
                args.repeats,
            ),
            "DP divide-conquer O(knlogn)": benchmark_gpu(
                lambda: weighted_triton_full(
                    lambda values, weights: dp_divide_conquer_triton(
                        values, codebook_size, weights=weights
                    )
                ),
                args.repeats,
            ),
            "DP D&C parallel scan": benchmark_gpu(
                lambda: weighted_triton_full(
                    lambda values, weights: dp_divide_conquer_parallel_scan_triton(
                        values, codebook_size, weights=weights
                    )
                ),
                args.repeats,
            ),
            "DP D&C fused stages": benchmark_gpu(
                lambda: weighted_triton_full(
                    lambda values, weights: dp_divide_conquer_fused_stages_triton(
                        values, codebook_size, weights=weights
                    )
                ),
                args.repeats,
            ),
            "DP SMAWK O(kn)": benchmark_gpu(
                lambda: weighted_triton_full(
                    lambda values, weights: dp_smawk_triton(
                        values, codebook_size, weights=weights
                    )
                ),
                args.repeats,
            ),
        }
        print(
            "Weighted benchmark (column-RMS proxy; no calibration activations; "
            f"importance range {column_importance.min().item():.3f}-"
            f"{column_importance.max().item():.3f}):"
        )
        print(
            f"{'Method':32s} {'ms':>10s} {'Mweights/s':>13s} "
            f"{'weighted MSE':>14s} {'MSE/signal':>13s}"
        )
        for name, latency in weighted_full_timings.items():
            throughput = weight.numel() / (latency * 1000.0)
            print(
                f"{name:32s} {latency:10.3f} {throughput:13.1f} "
                f"{weighted_mse[name]:14.6g} "
                f"{weighted_mse[name] / weighted_signal * 100:12.3f}%"
            )
        print(f"Packed weighted CT output: {weighted_ct_packed.numel():,} bytes")


if __name__ == "__main__":
    main()
