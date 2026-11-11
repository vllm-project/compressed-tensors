# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact weighted or unweighted 1D k-means fitting for LUT-B codebooks."""

import torch
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import tl, triton, triton_req


def _lut_b_dp_triton_req(
    values: torch.Tensor,
    num_centers: int,
    weights: torch.Tensor | None = None,
) -> bool:
    """Check the shape and device limits supported by the Triton DP kernel."""
    return (
        triton_req(values)
        and values.ndim == 2
        and values.shape[0] > 0
        and values.shape[1] <= 1024
        and 1 <= num_centers < values.shape[1]
        and (
            weights is None
            or (
                weights.shape == values.shape
                and weights.device == values.device
                and weights.dtype == torch.float32
                and weights.is_contiguous()
            )
        )
    )


@triton.jit
def _bitonic_sort_pairs(x, weights, indices, LOG_N: tl.constexpr):
    """Sort values and their weights together for the weighted prefix sums."""
    for stage in tl.static_range(1, LOG_N + 1):
        for bit in tl.static_range(stage - 1, -1, -1):
            partner_indices = indices ^ (1 << bit)
            partner_x = tl.gather(x, partner_indices, axis=0)
            partner_weights = tl.gather(weights, partner_indices, axis=0)
            ascending = (indices & (1 << stage)) == 0
            right_half = (indices & (1 << bit)) != 0
            keep_self = tl.where(
                ascending,
                tl.where(right_half, x >= partner_x, x <= partner_x),
                tl.where(right_half, x <= partner_x, x >= partner_x),
            )
            x = tl.where(keep_self, x, partner_x)
            weights = tl.where(keep_self, weights, partner_weights)
    return x, weights


@triton.jit
def _lut_b_dp_prefix_kernel(
    X,
    WEIGHTS,
    PREFIX_0,
    PREFIX_1,
    PREFIX_2,
    MEANS,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG_N: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
):
    """Sort one tile and create the weighted segment-cost prefixes."""
    tile = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid = cols < N
    x = tl.load(X + tile * N + cols, valid, other=0.0).to(tl.float32)
    if HAS_WEIGHTS:
        weights = tl.load(WEIGHTS + tile * N + cols, valid, other=0.0).to(tl.float32)
        x = tl.where(valid, x, float("inf"))
        x, weights = _bitonic_sort_pairs(x, weights, cols, LOG_N)
    else:
        x = tl.sort(tl.where(valid, x, float("inf")), descending=False)
        weights = tl.where(valid, 1.0, 0.0)

    safe_x = tl.where(valid, x, 0.0)
    total_weight = tl.sum(weights, 0)
    mean = tl.sum(weights * safe_x, 0) / tl.maximum(total_weight, 1.0e-30)
    centered = safe_x - mean
    weighted_centered = weights * centered
    p0 = tl.cumsum(weights, 0)
    p1 = tl.cumsum(weighted_centered, 0)
    p2 = tl.cumsum(weighted_centered * centered, 0)
    stride = N + 1
    tl.store(PREFIX_0 + tile * stride, 0.0)
    tl.store(PREFIX_1 + tile * stride, 0.0)
    tl.store(PREFIX_2 + tile * stride, 0.0)
    tl.store(PREFIX_0 + tile * stride + cols + 1, p0, valid)
    tl.store(PREFIX_1 + tile * stride + cols + 1, p1, valid)
    tl.store(PREFIX_2 + tile * stride + cols + 1, p2, valid)
    tl.store(MEANS + tile, mean)


@triton.jit
def _lut_b_dp_centers_kernel(
    PREFIX_0, PREFIX_1, MEANS, ARGS, CENTERS, N: tl.constexpr, K: tl.constexpr
):
    tile = tl.program_id(0)
    cluster = tl.program_id(1)
    stride = N + 1
    arg_stride = (K + 1) * stride
    end = N
    stage = K
    while stage > cluster + 1:
        end = tl.load(ARGS + tile * arg_stride + stage * stride + end).to(tl.int32)
        stage -= 1
    start = tl.load(ARGS + tile * arg_stride + (cluster + 1) * stride + end).to(
        tl.int32
    )
    mass = tl.load(PREFIX_0 + tile * stride + end) - tl.load(
        PREFIX_0 + tile * stride + start
    )
    delta = tl.load(PREFIX_1 + tile * stride + end) - tl.load(
        PREFIX_1 + tile * stride + start
    )
    center = tl.where(mass > 0.0, delta / tl.maximum(mass, 1.0e-30), 0.0)
    center += tl.load(MEANS + tile)
    tl.store(CENTERS + tile * K + cluster, center)


@triton.jit
def _lut_b_dp_fused_stages_kernel(
    PREFIX_0,
    PREFIX_1,
    PREFIX_2,
    DP,
    ARGS,
    TASKS,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_STACK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Run all DP rows for one tile with a parallel candidate scan."""
    tile = tl.program_id(0)
    stride = N + 1
    task_base = tile * MAX_STACK * 4
    init_indices = tl.arange(0, BLOCK_N)
    first_dp_base = DP + tile * 2 * stride
    tl.store(
        first_dp_base + init_indices,
        tl.where(init_indices == 0, 0.0, float("inf")),
        init_indices <= N,
    )

    scan_offsets = tl.arange(0, BLOCK_I)
    stage = 1
    while stage <= K:
        d_prev = first_dp_base + ((stage - 1) % 2) * stride
        d_next = first_dp_base + (stage % 2) * stride
        args_stage = ARGS + tile * (K + 1) * stride + stage * stride
        top = 0
        tl.store(TASKS + task_base + 0, stage)
        tl.store(TASKS + task_base + 1, N)
        tl.store(TASKS + task_base + 2, stage - 1)
        tl.store(TASKS + task_base + 3, N - 1)

        while top >= 0:
            task_offset = task_base + top * 4
            left = tl.load(TASKS + task_offset + 0)
            right = tl.load(TASKS + task_offset + 1)
            opt_left = tl.load(TASKS + task_offset + 2)
            opt_right = tl.load(TASKS + task_offset + 3)
            top -= 1

            if left <= right:
                mid = (left + right) // 2
                candidate_end = tl.minimum(opt_right, mid - 1)
                scan_start = opt_left
                best_split = opt_left
                best_cost = float("inf")
                p0_end = tl.load(PREFIX_0 + tile * stride + mid)
                p1_end = tl.load(PREFIX_1 + tile * stride + mid)
                p2_end = tl.load(PREFIX_2 + tile * stride + mid)
                while scan_start <= candidate_end:
                    candidate_ids = scan_start + scan_offsets
                    valid = (candidate_ids <= candidate_end) & (candidate_ids < mid)
                    p0_start = tl.load(
                        PREFIX_0 + tile * stride + candidate_ids,
                        valid,
                        other=0.0,
                    )
                    p1_start = tl.load(
                        PREFIX_1 + tile * stride + candidate_ids,
                        valid,
                        other=0.0,
                    )
                    p2_start = tl.load(
                        PREFIX_2 + tile * stride + candidate_ids,
                        valid,
                        other=0.0,
                    )
                    mass = p0_end - p0_start
                    delta = p1_end - p1_start
                    segment_cost = tl.where(
                        mass > 0.0,
                        p2_end - p2_start - delta * delta / tl.maximum(mass, 1.0e-30),
                        0.0,
                    )
                    previous = tl.load(
                        d_prev + candidate_ids,
                        valid,
                        other=float("inf"),
                    )
                    total = tl.where(valid, previous + segment_cost, float("inf"))
                    chunk_index = tl.argmin(total, 0)
                    chunk_cost = tl.min(total, 0)
                    if chunk_cost < best_cost:
                        best_cost = chunk_cost
                        best_split = scan_start + chunk_index
                    scan_start += BLOCK_I

                tl.store(d_next + mid, best_cost)
                tl.store(args_stage + mid, best_split)

                if mid < right:
                    top += 1
                    task_offset = task_base + top * 4
                    tl.store(TASKS + task_offset + 0, mid + 1)
                    tl.store(TASKS + task_offset + 1, right)
                    tl.store(TASKS + task_offset + 2, best_split)
                    tl.store(TASKS + task_offset + 3, opt_right)
                if left < mid:
                    top += 1
                    task_offset = task_base + top * 4
                    tl.store(TASKS + task_offset + 0, left)
                    tl.store(TASKS + task_offset + 1, mid - 1)
                    tl.store(TASKS + task_offset + 2, opt_left)
                    tl.store(TASKS + task_offset + 3, best_split)
        stage += 1


@ImplBackend.register("fit_lut_b_codebooks_dp", _lut_b_dp_triton_req, 0)
def _fit_lut_b_codebooks_dp_triton(
    values: torch.Tensor,
    num_centers: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton implementation shared by weighted and unweighted fitting."""
    values = values.to(torch.float32).contiguous()
    has_weights = weights is not None
    weight_ptr = values if weights is None else weights.contiguous()
    tiles, n = values.shape
    stride = n + 1
    block_n = triton.next_power_of_2(n)
    prefix_0 = torch.empty((tiles, stride), device=values.device, dtype=torch.float32)
    prefix_1 = torch.empty_like(prefix_0)
    prefix_2 = torch.empty_like(prefix_0)
    means = torch.empty((tiles,), device=values.device, dtype=torch.float32)
    _lut_b_dp_prefix_kernel[(tiles,)](
        values,
        weight_ptr,
        prefix_0,
        prefix_1,
        prefix_2,
        means,
        n,
        block_n,
        block_n.bit_length() - 1,
        has_weights,
        num_warps=4,
    )

    dp = torch.empty((tiles, 2, stride), device=values.device, dtype=torch.float32)
    args = torch.empty(
        (tiles, num_centers + 1, stride), device=values.device, dtype=torch.int32
    )
    max_stack = 2 * n.bit_length() + 4
    tasks = torch.empty((tiles, max_stack, 4), device=values.device, dtype=torch.int32)
    _lut_b_dp_fused_stages_kernel[(tiles,)](
        prefix_0,
        prefix_1,
        prefix_2,
        dp,
        args,
        tasks,
        n,
        num_centers,
        max_stack,
        triton.next_power_of_2(stride),
        32,
        num_warps=1,
    )

    centers = torch.empty(
        (tiles, num_centers), device=values.device, dtype=torch.float32
    )
    _lut_b_dp_centers_kernel[(tiles, num_centers)](
        prefix_0, prefix_1, means, args, centers, n, num_centers, num_warps=1
    )
    return centers


def _fit_lut_b_codebooks_dp_eager(
    values: torch.Tensor,
    num_centers: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch fallback for exact weighted or unweighted 1D k-means DP."""
    values = values.to(torch.float32)
    tiles, n = values.shape
    if num_centers >= n:
        if n == 0:
            return torch.zeros(
                (tiles, num_centers), device=values.device, dtype=torch.float32
            )
        sorted_values = values.sort(dim=1).values
        if num_centers == n:
            return sorted_values
        padding = sorted_values[:, -1:].expand(-1, num_centers - n)
        return torch.cat((sorted_values, padding), dim=1)

    if weights is None:
        sorted_values = values.sort(dim=1).values
        sorted_weights = None
    else:
        sorted_values, order = values.sort(dim=1)
        sorted_weights = weights.gather(1, order)

    centers = torch.empty(
        (tiles, num_centers), device=values.device, dtype=torch.float32
    )
    tiny = torch.finfo(torch.float32).tiny

    for tile in range(tiles):
        x = sorted_values[tile]
        tile_weights = (
            torch.ones_like(x) if sorted_weights is None else sorted_weights[tile]
        )
        total_weight = tile_weights.sum()
        mean = (tile_weights * x).sum() / total_weight.clamp_min(tiny)
        centered = x - mean
        prefix_0 = torch.cat((tile_weights.new_zeros(1), tile_weights.cumsum(0)))
        prefix_1 = torch.cat(
            (centered.new_zeros(1), (tile_weights * centered).cumsum(0))
        )
        prefix_2 = torch.cat(
            (centered.new_zeros(1), (tile_weights * centered.square()).cumsum(0))
        )
        previous = torch.full(
            (n + 1,), float("inf"), device=x.device, dtype=torch.float32
        )
        previous[0] = 0.0
        backpointers = torch.full(
            (num_centers + 1, n + 1), -1, device=x.device, dtype=torch.int32
        )

        for stage in range(1, num_centers + 1):
            current = torch.full_like(previous, float("inf"))

            def solve(left: int, right: int, opt_left: int, opt_right: int):
                if left > right:
                    return
                mid = (left + right) // 2
                candidate_end = min(opt_right, mid - 1)
                candidates = torch.arange(
                    opt_left, candidate_end + 1, device=x.device, dtype=torch.long
                )
                mass = prefix_0[mid] - prefix_0[candidates]
                delta = prefix_1[mid] - prefix_1[candidates]
                cost = (
                    prefix_2[mid]
                    - prefix_2[candidates]
                    - delta.square() / mass.clamp_min(tiny)
                )
                segment_cost = torch.where(mass > 0.0, cost, 0.0)
                candidate_cost = previous[candidates] + segment_cost
                best_index = int(torch.argmin(candidate_cost).item())
                best_split = opt_left + best_index
                current[mid] = candidate_cost[best_index]
                backpointers[stage, mid] = best_split
                solve(left, mid - 1, opt_left, best_split)
                solve(mid + 1, right, best_split, opt_right)

            solve(stage, n, stage - 1, n - 1)
            previous = current

        end = n
        for stage in range(num_centers, 0, -1):
            start = int(backpointers[stage, end].item())
            mass = prefix_0[end] - prefix_0[start]
            delta = prefix_1[end] - prefix_1[start]
            centers[tile, stage - 1] = (
                torch.where(mass > 0.0, delta / mass.clamp_min(tiny), 0.0) + mean
            )
            end = start

    return centers


@ImplBackend.entrypoint("fit_lut_b_codebooks_dp")
def fit_lut_b_codebooks_dp(
    values: torch.Tensor,
    num_centers: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit exact 1D k-means codebooks, dispatching to Triton when available."""
    if values.ndim != 2:
        raise ValueError(f"LUT-B DP expects [tiles, values] input, got {values.shape}")
    if num_centers < 1:
        raise ValueError(f"num_centers must be positive, got {num_centers}")
    if weights is not None:
        if weights.shape != values.shape:
            raise ValueError(
                "LUT-B DP weights must match values "
                f"{values.shape}, got {weights.shape}"
            )
        if not torch.isfinite(weights).all():
            raise ValueError("LUT-B DP weights must be finite")
        if torch.any(weights < 0):
            raise ValueError("LUT-B DP weights must be non-negative")
    return _fit_lut_b_codebooks_dp_eager(values, num_centers, weights)
