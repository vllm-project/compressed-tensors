"""Compare Triton LUT-B Lloyd, 1D k-means DP variants, and e-means.

The exact k-means implementations include exhaustive O(k n^2) DP, monotone
divide-and-conquer O(k n log n) DP with scalar or parallel candidate scans, and
SMAWK O(k n) DP for sorted tile values.
E-means uses the project's scalar-data automatic recipe and no-revival default.
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# Fitted e-means recipe shipped by the project. The automatic policy uses the
# quality batch law for scalar LUT-B tiles (d=1); see emeans_recipe below.
_EMEANS_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}
_EMEANS_STEP_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def _round_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


def emeans_recipe(
    n: int, k: int, epochs: int, d: int = 1, batch_law: str = "auto"
):
    """Resolve the project scaling laws for the LUT-B clustering shape."""
    if batch_law == "auto":
        law = "quality" if d <= 16 else "throughput"
    elif batch_law in _EMEANS_BATCH_LAWS:
        law = batch_law
    else:
        raise ValueError(f"unsupported e-means batch law: {batch_law}")
    coefficient, k_exp, n_exp, epoch_exp = _EMEANS_BATCH_LAWS[law]
    raw_batch = coefficient * k**k_exp * n**n_exp * epochs**epoch_exp
    batch_size = _round_power_of_two(raw_batch)
    # The project caps quality batches at n/2 and throughput batches at the
    # nearest power of two to n/2.
    if law == "quality":
        batch_size = min(batch_size, n // 2)
    else:
        batch_size = min(batch_size, _round_power_of_two(n / 2))
    batch_size = max(1, min(batch_size, n))

    alpha_c, alpha_k, alpha_b, alpha_n, alpha_t = _EMEANS_STEP_LAW
    alpha0 = min(
        max(
            alpha_c
            * k**alpha_k
            * batch_size**alpha_b
            * n**alpha_n
            * epochs**alpha_t,
            1e-4,
        ),
        1.0,
    )
    return law, batch_size, alpha0


@triton.jit
def _snap_e4m3(values):
    values = tl.minimum(tl.maximum(values, -448.0), 448.0)
    return values.to(tl.float8e4nv).to(tl.float32)


@triton.jit
def _normalize_tiles_kernel(X, Y, N: tl.constexpr, BLOCK_N: tl.constexpr):
    tile = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + tile * N + cols, cols < N, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1.17549435e-38)
    exponent = tl.ceil(tl.log2(amax / 448.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    normalized = x * tl.exp2(-exponent)
    tl.store(Y + tile * N + cols, normalized, cols < N)


@triton.jit
def _lloyd_kernel(
    X,
    CENTERS,
    N: tl.constexpr,
    K: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    center_ids = tl.arange(0, K)
    x = tl.load(X + tile * N + cols, cols < N, other=0.0).to(tl.float32)

    # Match CT's mean-plus-farthest-point initialization for unit importance.
    mean = tl.sum(x, 0) / N
    centers = tl.full((K,), 0.0, tl.float32)
    centers = tl.where(center_ids == 0, mean, centers)
    min_dist = (x - mean) * (x - mean)
    for center_id in tl.static_range(1, K):
        farthest = tl.argmax(min_dist, 0)
        new_center = tl.sum(tl.where(cols == farthest, x, 0.0), 0)
        centers = tl.where(center_ids == center_id, new_center, centers)
        dist = (x - new_center) * (x - new_center)
        min_dist = tl.minimum(min_dist, dist)

    centers = tl.sort(_snap_e4m3(centers), descending=False)
    for _ in range(ITERATIONS):
        distances = x[:, None] - centers[None, :]
        assignments = tl.argmin(distances * distances, 1)
        belongs = assignments[None, :] == center_ids[:, None]
        sums = tl.sum(tl.where(belongs, x[None, :], 0.0), 1)
        counts = tl.sum(belongs.to(tl.float32), 1)
        updated = tl.where(counts > 0.0, sums / tl.maximum(counts, 1.0), centers)
        centers = tl.sort(_snap_e4m3(updated), descending=False)

    tl.store(CENTERS + tile * K + center_ids, centers)


@triton.jit
def _emeans_train_kernel(
    X,
    PERMUTATIONS,
    STATE_CENTERS,
    LAST_LABELS,
    N: tl.constexpr,
    K: tl.constexpr,
    EPOCHS: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    STEPS_PER_EPOCH: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
    ALPHA0: tl.constexpr,
):
    tile = tl.program_id(0)
    center_ids = tl.arange(0, K)
    batch_ids = tl.arange(0, BATCH_SIZE)
    permutation_stride = EPOCHS * N

    # Random data-point initialization, as used by e-means.
    initial_ids = tl.load(
        PERMUTATIONS + tile * permutation_stride + center_ids
    ).to(tl.int32)
    centers = tl.load(X + tile * N + initial_ids).to(tl.float32)
    usage = tl.full((K,), 1.52587890625e-5, tl.float32)  # 2**-16

    for step in range(TOTAL_STEPS):
        epoch = step // STEPS_PER_EPOCH
        batch = step % STEPS_PER_EPOCH
        permutation_offset = (
            tile * permutation_stride + epoch * N + batch * BATCH_SIZE
        )
        ids = tl.load(
            PERMUTATIONS + permutation_offset + batch_ids,
            batch_ids < BATCH_SIZE,
            other=0,
        ).to(tl.int32)
        x = tl.load(X + tile * N + ids, batch_ids < BATCH_SIZE, other=0.0)
        distances = x[:, None] - centers[None, :]
        assignments = tl.argmin(distances * distances, 1)
        belongs = assignments[None, :] == center_ids[:, None]
        batch_counts = tl.sum(belongs.to(tl.float32), 1)
        batch_sums = tl.sum(tl.where(belongs, x[None, :], 0.0), 1)

        progress = step.to(tl.float32) / TOTAL_STEPS
        cosine = 0.5 * (1.0 + tl.cos(3.141592653589793 * progress))
        alpha = ALPHA0 * (0.1 + 0.9 * cosine)
        next_usage = (1.0 - alpha) * usage + alpha * batch_counts
        next_sums = (1.0 - alpha) * usage * centers + alpha * batch_sums
        centers = tl.where(
            next_usage > 0.0,
            next_sums / tl.maximum(next_usage, 1.0e-30),
            centers,
        )
        usage = next_usage

        if epoch == EPOCHS - 1:
            tl.store(LAST_LABELS + tile * N + ids, assignments)

    tl.store(STATE_CENTERS + tile * K + center_ids, centers)


@triton.jit
def _emeans_finalize_kernel(
    X,
    STATE_CENTERS,
    LAST_LABELS,
    CENTERS,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    center_ids = tl.arange(0, K)
    x = tl.load(X + tile * N + cols, cols < N, other=0.0).to(tl.float32)
    labels = tl.load(LAST_LABELS + tile * N + cols, cols < N, other=-1)
    belongs = labels[None, :] == center_ids[:, None]
    counts = tl.sum(belongs.to(tl.float32), 1)
    sums = tl.sum(tl.where(belongs, x[None, :], 0.0), 1)
    running = tl.load(STATE_CENTERS + tile * K + center_ids)
    means = tl.where(counts > 0.0, sums / tl.maximum(counts, 1.0), running)
    centers = tl.sort(_snap_e4m3(means), descending=False)
    tl.store(CENTERS + tile * K + center_ids, centers)


@triton.jit
def _bitonic_sort_pairs(x, weights, indices, LOG_N: tl.constexpr):
    """Sort x and carry its per-point weights through the same network."""
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
def _dp_prefix_kernel(
    X,
    WEIGHTS,
    PREFIX_0,
    PREFIX_1,
    PREFIX_2,
    MEANS,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG_N: tl.constexpr,
    SORT_INPUT: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
):
    tile = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid = cols < N
    x = tl.load(X + tile * N + cols, valid, other=0.0).to(tl.float32)
    if HAS_WEIGHTS:
        weights = tl.load(WEIGHTS + tile * N + cols, valid, other=0.0).to(tl.float32)
    else:
        weights = tl.where(valid, 1.0, 0.0)
    if SORT_INPUT:
        x = tl.where(valid, x, float("inf"))
        if HAS_WEIGHTS:
            x, weights = _bitonic_sort_pairs(x, weights, cols, LOG_N)
        else:
            x = tl.sort(x, descending=False)
    safe_x = tl.where(valid, x, 0.0)
    total_weight = tl.sum(weights, 0)
    weighted_mean = tl.sum(weights * safe_x, 0) / tl.maximum(total_weight, 1.0e-30)
    centered = safe_x - weighted_mean
    weighted_centered = weights * centered
    p0 = tl.cumsum(weights, 0)
    p1 = tl.cumsum(weighted_centered, 0)
    p2 = tl.cumsum(weighted_centered * centered, 0)
    prefix_stride = N + 1
    tl.store(PREFIX_0 + tile * prefix_stride, 0.0)
    tl.store(PREFIX_1 + tile * prefix_stride, 0.0)
    tl.store(PREFIX_2 + tile * prefix_stride, 0.0)
    tl.store(PREFIX_0 + tile * prefix_stride + cols + 1, p0, valid)
    tl.store(PREFIX_1 + tile * prefix_stride + cols + 1, p1, valid)
    tl.store(PREFIX_2 + tile * prefix_stride + cols + 1, p2, valid)
    tl.store(MEANS + tile, weighted_mean)


@triton.jit
def _sort_tiles_kernel(X, Y, N: tl.constexpr, BLOCK_N: tl.constexpr):
    tile = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + tile * N + cols, cols < N, other=0.0).to(tl.float32)
    x = tl.sort(x, descending=False)
    tl.store(Y + tile * N + cols, x, cols < N)


@triton.jit
def _dp_stage_kernel(
    PREFIX_0,
    PREFIX_1,
    PREFIX_2,
    D_PREV,
    D_NEXT,
    ARGS,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    tile = tl.program_id(0)
    end = tl.program_id(1)
    i = tl.arange(0, BLOCK_I)
    stride = N + 1
    w_end = tl.load(PREFIX_0 + tile * stride + end)
    p1_end = tl.load(PREFIX_1 + tile * stride + end)
    p2_end = tl.load(PREFIX_2 + tile * stride + end)
    w_start = tl.load(PREFIX_0 + tile * stride + i, i <= N, other=0.0)
    p1_start = tl.load(PREFIX_1 + tile * stride + i, i <= N, other=0.0)
    p2_start = tl.load(PREFIX_2 + tile * stride + i, i <= N, other=0.0)
    previous = tl.load(D_PREV + tile * stride + i, i <= N, other=float("inf"))
    mass = w_end - w_start
    valid = (i < end) & (i <= N)
    safe_mass = tl.maximum(mass, 1.0e-30)
    s1 = p1_end - p1_start
    s2 = p2_end - p2_start
    segment_cost = tl.where(mass > 0.0, s2 - s1 * s1 / safe_mass, 0.0)
    total = tl.where(valid, previous + segment_cost, float("inf"))
    best_i = tl.argmin(total, 0)
    best_cost = tl.min(total, 0)
    tl.store(D_NEXT + tile * stride + end, best_cost)
    tl.store(ARGS + tile * (K + 1) * stride + M * stride + end, best_i)


@triton.jit
def _dp_init_kernel(D, N: tl.constexpr, BLOCK_N: tl.constexpr):
    tile = tl.program_id(0)
    indices = tl.arange(0, BLOCK_N)
    valid = indices <= N
    tl.store(D + tile * (N + 1) + indices, 0.0, valid & (indices == 0))
    tl.store(D + tile * (N + 1) + indices, float("inf"), valid & (indices > 0))


@triton.jit
def _dp_matrix_value(
    PREFIX_0, PREFIX_1, PREFIX_2, D_PREV, tile, row, col, N: tl.constexpr
):
    stride = N + 1
    if col < row:
        w_end = tl.load(PREFIX_0 + tile * stride + row)
        w_start = tl.load(PREFIX_0 + tile * stride + col)
        p1_end = tl.load(PREFIX_1 + tile * stride + row)
        p2_end = tl.load(PREFIX_2 + tile * stride + row)
        p1_start = tl.load(PREFIX_1 + tile * stride + col)
        p2_start = tl.load(PREFIX_2 + tile * stride + col)
        mass = w_end - w_start
        delta = p1_end - p1_start
        cost = tl.where(
            mass > 0.0,
            p2_end - p2_start - delta * delta / tl.maximum(mass, 1.0e-30),
            0.0,
        )
        prior = tl.load(D_PREV + tile * stride + col)
        return prior + cost
    else:
        # SMAWK's rectangular extension represents an empty final cluster.
        return tl.load(D_PREV + tile * stride + row)


@triton.jit
def _dp_dc_stage_kernel(
    PREFIX_0,
    PREFIX_1,
    PREFIX_2,
    D_PREV,
    D_NEXT,
    ARGS,
    TASKS,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    MAX_STACK: tl.constexpr,
):
    """One DP row via monotone divide-and-conquer, O(n log n) probes."""
    tile = tl.program_id(0)
    stride = N + 1
    task_base = tile * MAX_STACK * 4
    top = 0
    tl.store(TASKS + task_base + 0, M)
    tl.store(TASKS + task_base + 1, N)
    tl.store(TASKS + task_base + 2, M - 1)
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
            candidate = opt_left
            candidate_end = tl.minimum(opt_right, mid - 1)
            best_cost = float("inf")
            best_split = opt_left
            while candidate <= candidate_end:
                value = _dp_matrix_value(
                    PREFIX_0,
                    PREFIX_1,
                    PREFIX_2,
                    D_PREV,
                    tile,
                    mid,
                    candidate,
                    N,
                )
                better = value < best_cost
                best_cost = tl.minimum(best_cost, value)
                best_split = tl.where(better, candidate, best_split)
                candidate += 1

            tl.store(D_NEXT + tile * stride + mid, best_cost)
            tl.store(
                ARGS + tile * (K + 1) * stride + M * stride + mid,
                best_split,
            )

            # Push right first so the left interval is processed next.
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


@triton.jit
def _dp_dc_parallel_scan_stage_kernel(
    PREFIX_0,
    PREFIX_1,
    PREFIX_2,
    D_PREV,
    D_NEXT,
    ARGS,
    TASKS,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    MAX_STACK: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """D&C DP row with each candidate scan evaluated in parallel."""
    tile = tl.program_id(0)
    stride = N + 1
    scan_offsets = tl.arange(0, BLOCK_I)
    task_base = tile * MAX_STACK * 4
    top = 0
    tl.store(TASKS + task_base + 0, M)
    tl.store(TASKS + task_base + 1, N)
    tl.store(TASKS + task_base + 2, M - 1)
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
                    PREFIX_0 + tile * stride + candidate_ids, valid, other=0.0
                )
                p1_start = tl.load(
                    PREFIX_1 + tile * stride + candidate_ids, valid, other=0.0
                )
                p2_start = tl.load(
                    PREFIX_2 + tile * stride + candidate_ids, valid, other=0.0
                )
                mass = p0_end - p0_start
                delta = p1_end - p1_start
                segment_cost = tl.where(
                    mass > 0.0,
                    p2_end - p2_start - delta * delta / tl.maximum(mass, 1.0e-30),
                    0.0,
                )
                previous = tl.load(
                    D_PREV + tile * stride + candidate_ids,
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

            tl.store(D_NEXT + tile * stride + mid, best_cost)
            tl.store(
                ARGS + tile * (K + 1) * stride + M * stride + mid,
                best_split,
            )

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


@triton.jit
def _dp_dc_local_stack_stage_kernel(
    PREFIX_0,
    PREFIX_1,
    PREFIX_2,
    D_PREV,
    D_NEXT,
    ARGS,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_STACK: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Parallel candidate scan with the recursion stack held CTA-locally."""
    tile = tl.program_id(0)
    stride = N + 1
    stack_ids = tl.arange(0, BLOCK_STACK)
    scan_offsets = tl.arange(0, BLOCK_I)
    stack_left = tl.full((BLOCK_STACK,), 0, tl.int32)
    stack_right = tl.full((BLOCK_STACK,), 0, tl.int32)
    stack_opt_left = tl.full((BLOCK_STACK,), 0, tl.int32)
    stack_opt_right = tl.full((BLOCK_STACK,), 0, tl.int32)
    top = 0
    stack_left = tl.where(stack_ids == 0, M, stack_left)
    stack_right = tl.where(stack_ids == 0, N, stack_right)
    stack_opt_left = tl.where(stack_ids == 0, M - 1, stack_opt_left)
    stack_opt_right = tl.where(stack_ids == 0, N - 1, stack_opt_right)

    while top >= 0:
        left = tl.sum(tl.where(stack_ids == top, stack_left, 0), 0)
        right = tl.sum(tl.where(stack_ids == top, stack_right, 0), 0)
        opt_left = tl.sum(tl.where(stack_ids == top, stack_opt_left, 0), 0)
        opt_right = tl.sum(tl.where(stack_ids == top, stack_opt_right, 0), 0)
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
                    PREFIX_0 + tile * stride + candidate_ids, valid, other=0.0
                )
                p1_start = tl.load(
                    PREFIX_1 + tile * stride + candidate_ids, valid, other=0.0
                )
                p2_start = tl.load(
                    PREFIX_2 + tile * stride + candidate_ids, valid, other=0.0
                )
                mass = p0_end - p0_start
                delta = p1_end - p1_start
                segment_cost = tl.where(
                    mass > 0.0,
                    p2_end - p2_start - delta * delta / tl.maximum(mass, 1.0e-30),
                    0.0,
                )
                previous = tl.load(
                    D_PREV + tile * stride + candidate_ids,
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

            tl.store(D_NEXT + tile * stride + mid, best_cost)
            tl.store(
                ARGS + tile * (K + 1) * stride + M * stride + mid,
                best_split,
            )

            if mid < right:
                top += 1
                stack_left = tl.where(stack_ids == top, mid + 1, stack_left)
                stack_right = tl.where(stack_ids == top, right, stack_right)
                stack_opt_left = tl.where(stack_ids == top, best_split, stack_opt_left)
                stack_opt_right = tl.where(stack_ids == top, opt_right, stack_opt_right)
            if left < mid:
                top += 1
                stack_left = tl.where(stack_ids == top, left, stack_left)
                stack_right = tl.where(stack_ids == top, mid - 1, stack_right)
                stack_opt_left = tl.where(stack_ids == top, opt_left, stack_opt_left)
                stack_opt_right = tl.where(stack_ids == top, best_split, stack_opt_right)


@triton.jit
def _dp_dc_fused_stages_kernel(
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
    """Run all D&C DP rows for one tile in a single CTA launch."""
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
                        d_prev + candidate_ids, valid, other=float("inf")
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


@triton.jit
def _dp_smawk_stage_kernel(
    PREFIX_0,
    PREFIX_1,
    PREFIX_2,
    D_PREV,
    D_NEXT,
    ARGS,
    COLS,
    COL_COUNTS,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    MAX_LEVELS: tl.constexpr,
):
    """One DP row using SMAWK, O(n) implicit-matrix probes."""
    tile = tl.program_id(0)
    stride = N + 1
    row_count_0 = N - M + 1
    row_count = row_count_0
    row_start = M
    row_stride = 1
    col_count = row_count_0
    level = 0
    cols_tile_base = COLS + tile * MAX_LEVELS * stride
    counts_tile_base = COL_COUNTS + tile * MAX_LEVELS

    # Reduce columns at each recursive level, then recurse on odd-index rows.
    while row_count > 0:
        out_base = cols_tile_base + level * stride
        top = 0
        col_pos = 0
        while col_pos < col_count:
            if level == 0:
                candidate_col = M - 1 + col_pos
            else:
                prev_base = cols_tile_base + (level - 1) * stride
                candidate_col = tl.load(prev_base + col_pos)

            keep_popping = 1
            while (top > 0) & (keep_popping != 0):
                compare_row = row_start + (top - 1) * row_stride
                last_col = tl.load(out_base + top - 1)
                candidate_value = _dp_matrix_value(
                    PREFIX_0,
                    PREFIX_1,
                    PREFIX_2,
                    D_PREV,
                    tile,
                    compare_row,
                    candidate_col,
                    N,
                )
                last_value = _dp_matrix_value(
                    PREFIX_0,
                    PREFIX_1,
                    PREFIX_2,
                    D_PREV,
                    tile,
                    compare_row,
                    last_col,
                    N,
                )
                if candidate_value < last_value:
                    top -= 1
                else:
                    keep_popping = 0

            if top < row_count:
                tl.store(out_base + top, candidate_col)
                top += 1
            col_pos += 1

        col_count = top
        tl.store(counts_tile_base + level, col_count)
        if row_count == 1:
            row = row_start
            scan = 0
            best_cost = float("inf")
            best_col = tl.load(out_base)
            while scan < col_count:
                candidate_col = tl.load(out_base + scan)
                value = _dp_matrix_value(
                    PREFIX_0,
                    PREFIX_1,
                    PREFIX_2,
                    D_PREV,
                    tile,
                    row,
                    candidate_col,
                    N,
                )
                if value < best_cost:
                    best_cost = value
                    best_col = candidate_col
                scan += 1
            tl.store(D_NEXT + tile * stride + row, best_cost)
            tl.store(
                ARGS + tile * (K + 1) * stride + M * stride + row,
                best_col,
            )
            row_count = 0
        else:
            row_count = row_count // 2
            row_start += row_stride
            row_stride *= 2
            level += 1

    # Interpolate the even rows while unwinding the SMAWK recursion.
    parent_level = level - 1
    while parent_level >= 0:
        parent_rows = row_count_0 // (1 << parent_level)
        parent_start = M + (1 << parent_level) - 1
        parent_stride = 1 << parent_level
        parent_base = cols_tile_base + parent_level * stride
        parent_cols = tl.load(counts_tile_base + parent_level)
        even_pos = 0
        low_pos = 0
        while even_pos < parent_rows:
            row = parent_start + even_pos * parent_stride
            if even_pos + 1 < parent_rows:
                target_col = tl.load(
                    ARGS
                    + tile * (K + 1) * stride
                    + M * stride
                    + row
                    + parent_stride
                )
                high_pos = low_pos
                seek_high = 1
                while (high_pos < parent_cols) & (seek_high != 0):
                    value_col = tl.load(parent_base + high_pos)
                    if value_col >= target_col:
                        seek_high = 0
                    else:
                        high_pos += 1
                high_pos = tl.minimum(high_pos, parent_cols - 1)
            else:
                high_pos = parent_cols - 1

            scan = low_pos
            best_cost = float("inf")
            best_col = tl.load(parent_base + low_pos)
            while scan <= high_pos:
                candidate_col = tl.load(parent_base + scan)
                value = _dp_matrix_value(
                    PREFIX_0,
                    PREFIX_1,
                    PREFIX_2,
                    D_PREV,
                    tile,
                    row,
                    candidate_col,
                    N,
                )
                if value < best_cost:
                    best_cost = value
                    best_col = candidate_col
                scan += 1

            tl.store(D_NEXT + tile * stride + row, best_cost)
            tl.store(
                ARGS + tile * (K + 1) * stride + M * stride + row,
                best_col,
            )
            low_pos = high_pos
            even_pos += 2
        parent_level -= 1


@triton.jit
def _dp_centers_kernel(
    PREFIX_0,
    PREFIX_1,
    MEANS,
    ARGS,
    CENTERS,
    N: tl.constexpr,
    K: tl.constexpr,
):
    tile = tl.program_id(0)
    cluster = tl.program_id(1)
    stride = N + 1
    arg_stride = (K + 1) * stride
    end = N
    stage = K
    while stage > cluster + 1:
        end = tl.load(
            ARGS + tile * arg_stride + stage * stride + end
        ).to(tl.int32)
        stage -= 1
    start = tl.load(
        ARGS + tile * arg_stride + (cluster + 1) * stride + end
    ).to(tl.int32)
    w_end = tl.load(PREFIX_0 + tile * stride + end)
    w_start = tl.load(PREFIX_0 + tile * stride + start)
    p1_end = tl.load(PREFIX_1 + tile * stride + end)
    p1_start = tl.load(PREFIX_1 + tile * stride + start)
    mass = w_end - w_start
    mean = tl.where(
        mass > 0.0,
        (p1_end - p1_start) / tl.maximum(mass, 1.0e-30),
        0.0,
    ) + tl.load(MEANS + tile)
    center = _snap_e4m3(mean)
    tl.store(CENTERS + tile * K + cluster, center)


@triton.jit
def _dp_sort_centers_kernel(CENTERS, K: tl.constexpr, BLOCK_K: tl.constexpr):
    tile = tl.program_id(0)
    center_ids = tl.arange(0, BLOCK_K)
    centers = tl.load(CENTERS + tile * K + center_ids).to(tl.float32)
    centers = tl.sort(centers, descending=False)
    tl.store(CENTERS + tile * K + center_ids, centers)


@triton.jit
def _mse_kernel(
    X,
    CENTERS,
    MSE,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    center_ids = tl.arange(0, K)
    x = tl.load(X + tile * N + cols, cols < N, other=0.0).to(tl.float32)
    centers = tl.load(CENTERS + tile * K + center_ids).to(tl.float32)
    diff = x[:, None] - centers[None, :]
    errors = tl.min(diff * diff, 1)
    mse = tl.sum(tl.where(cols < N, errors, 0.0), 0) / N
    tl.store(MSE + tile, mse)


def lloyd_triton(values: torch.Tensor, k: int, iterations: int) -> torch.Tensor:
    tiles, n = values.shape
    centers = torch.empty((tiles, k), device=values.device, dtype=torch.float32)
    _lloyd_kernel[(tiles,)](
        values,
        centers,
        n,
        k,
        iterations,
        triton.next_power_of_2(n),
        num_warps=4,
    )
    return centers


def _prepare_dp_weights(
    values: torch.Tensor, weights: torch.Tensor | None
) -> torch.Tensor | None:
    if weights is None:
        return None
    weights = torch.as_tensor(
        weights, device=values.device, dtype=torch.float32
    )
    try:
        weights = torch.broadcast_to(weights, values.shape).contiguous()
    except RuntimeError as exc:
        raise ValueError(
            f"weight shape {tuple(weights.shape)} cannot broadcast to "
            f"tile values {tuple(values.shape)}"
        ) from exc
    if not torch.isfinite(weights).all():
        raise ValueError("k-means weights must be finite")
    if torch.any(weights < 0):
        raise ValueError("k-means weights must be non-negative")
    # A zero-mass tile has no defined weighted objective; use the usual
    # unweighted objective for that tile, matching LUT-B importance handling.
    total = weights.sum(dim=1, keepdim=True)
    weights = torch.where(total > 0, weights, torch.ones_like(weights))
    # A common positive scale per tile leaves its minimizer and centers intact.
    max_weight = weights.amax(dim=1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return (weights / max_weight).contiguous()


def _make_dp_prefixes(
    values: torch.Tensor,
    weights: torch.Tensor | None,
    presorted: bool,
):
    tiles, n = values.shape
    stride = n + 1
    prefix_0 = torch.empty((tiles, stride), device=values.device, dtype=torch.float32)
    prefix_1 = torch.empty((tiles, stride), device=values.device, dtype=torch.float32)
    prefix_2 = torch.empty_like(prefix_1)
    means = torch.empty((tiles,), device=values.device, dtype=torch.float32)
    weight_ptr = values if weights is None else weights
    _dp_prefix_kernel[(tiles,)](
        values,
        weight_ptr,
        prefix_0,
        prefix_1,
        prefix_2,
        means,
        n,
        triton.next_power_of_2(n),
        triton.next_power_of_2(n).bit_length() - 1,
        presorted is False,
        weights is not None,
        num_warps=4,
    )
    return prefix_0, prefix_1, prefix_2, means


def _dp_center_shortcut(values: torch.Tensor, k: int) -> torch.Tensor | None:
    """Return exact centers when every input point can have its own center."""
    tiles, n = values.shape
    if k < n:
        return None
    if n == 0:
        return torch.zeros((tiles, k), device=values.device, dtype=torch.float32)
    centers = values.sort(dim=1).values
    if k == n:
        return centers
    return torch.cat((centers, centers[:, -1:].expand(-1, k - n)), dim=1)


def dp_triton(
    values: torch.Tensor,
    k: int,
    *,
    weights: torch.Tensor | None = None,
    presorted: bool = False,
) -> torch.Tensor:
    """Exhaustive exact 1D k-means DP, optionally with per-point weights.

    ``weights`` must broadcast to ``values`` shaped ``[tiles, n]``. If
    ``presorted`` is true, values and weights must already be sorted together.
    """
    shortcut = _dp_center_shortcut(values, k)
    if shortcut is not None:
        return shortcut
    tiles, n = values.shape
    stride = n + 1
    has_weights = weights is not None
    weights = _prepare_dp_weights(values, weights)
    prefix_0, prefix_1, prefix_2, means = _make_dp_prefixes(
        values, weights, presorted
    )

    d_prev = torch.empty((tiles, stride), device=values.device, dtype=torch.float32)
    d_next = torch.empty_like(d_prev)
    _dp_init_kernel[(tiles,)](d_prev, n, triton.next_power_of_2(stride), num_warps=4)
    args = torch.empty(
        (tiles, k + 1, stride), device=values.device, dtype=torch.int32
    )
    grid = (tiles, stride)
    for stage in range(1, k + 1):
        _dp_stage_kernel[grid](
            prefix_0,
            prefix_1,
            prefix_2,
            d_prev,
            d_next,
            args,
            n,
            k,
            stage,
            triton.next_power_of_2(stride),
            num_warps=4,
        )
        d_prev, d_next = d_next, d_prev

    centers = torch.empty((tiles, k), device=values.device, dtype=torch.float32)
    _dp_centers_kernel[(tiles, k)](
        prefix_0, prefix_1, means, args, centers, n, k, num_warps=1
    )
    if has_weights:
        _dp_sort_centers_kernel[(tiles,)](
            centers, k, triton.next_power_of_2(k), num_warps=1
        )
    return centers


def dp_divide_conquer_triton(
    values: torch.Tensor,
    k: int,
    *,
    weights: torch.Tensor | None = None,
    presorted: bool = False,
) -> torch.Tensor:
    """Exact 1D k-means DP using monotone divide-and-conquer optimization.

    For sorted inputs this uses O(k n log n) segment-cost probes, versus the
    O(k n^2) exhaustive recurrence in :func:`dp_triton`. Optional nonnegative
    weights broadcast to ``values`` and are sorted together with each tile.
    With ``presorted=True``, values and weights must already be aligned/sorted.
    """
    shortcut = _dp_center_shortcut(values, k)
    if shortcut is not None:
        return shortcut
    tiles, n = values.shape
    stride = n + 1
    has_weights = weights is not None
    weights = _prepare_dp_weights(values, weights)
    prefix_0, prefix_1, prefix_2, means = _make_dp_prefixes(
        values, weights, presorted
    )

    d_prev = torch.empty((tiles, stride), device=values.device, dtype=torch.float32)
    d_next = torch.empty_like(d_prev)
    _dp_init_kernel[(tiles,)](
        d_prev, n, triton.next_power_of_2(stride), num_warps=4
    )
    args = torch.empty((tiles, k + 1, stride), device=values.device, dtype=torch.int32)
    max_stack = 2 * n.bit_length() + 4
    tasks = torch.empty(
        (tiles, max_stack, 4), device=values.device, dtype=torch.int32
    )

    for stage in range(1, k + 1):
        _dp_dc_stage_kernel[(tiles,)](
            prefix_0,
            prefix_1,
            prefix_2,
            d_prev,
            d_next,
            args,
            tasks,
            n,
            k,
            stage,
            max_stack,
            num_warps=1,
        )
        d_prev, d_next = d_next, d_prev

    centers = torch.empty((tiles, k), device=values.device, dtype=torch.float32)
    _dp_centers_kernel[(tiles, k)](
        prefix_0, prefix_1, means, args, centers, n, k, num_warps=1
    )
    if has_weights:
        _dp_sort_centers_kernel[(tiles,)](
            centers, k, triton.next_power_of_2(k), num_warps=1
        )
    return centers


def dp_divide_conquer_parallel_scan_triton(
    values: torch.Tensor,
    k: int,
    *,
    weights: torch.Tensor | None = None,
    presorted: bool = False,
    candidate_block_size: int = 32,
    backpointer_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """D&C DP with tunable candidate scans and backpointer storage width."""
    if candidate_block_size <= 0 or candidate_block_size & (candidate_block_size - 1):
        raise ValueError("candidate_block_size must be a positive power of two")
    shortcut = _dp_center_shortcut(values, k)
    if shortcut is not None:
        return shortcut
    tiles, n = values.shape
    if backpointer_dtype == torch.int16 and n > 32767:
        raise ValueError("int16 backpointers require tile length <= 32767")
    if backpointer_dtype not in (torch.int16, torch.int32):
        raise ValueError("backpointer_dtype must be torch.int16 or torch.int32")
    stride = n + 1
    has_weights = weights is not None
    weights = _prepare_dp_weights(values, weights)
    prefix_0, prefix_1, prefix_2, means = _make_dp_prefixes(
        values, weights, presorted
    )

    d_prev = torch.empty((tiles, stride), device=values.device, dtype=torch.float32)
    d_next = torch.empty_like(d_prev)
    _dp_init_kernel[(tiles,)](
        d_prev, n, triton.next_power_of_2(stride), num_warps=4
    )
    args = torch.empty(
        (tiles, k + 1, stride), device=values.device, dtype=backpointer_dtype
    )
    max_stack = 2 * n.bit_length() + 4
    tasks = torch.empty(
        (tiles, max_stack, 4), device=values.device, dtype=torch.int32
    )
    block_i = candidate_block_size

    for stage in range(1, k + 1):
        _dp_dc_parallel_scan_stage_kernel[(tiles,)](
            prefix_0,
            prefix_1,
            prefix_2,
            d_prev,
            d_next,
            args,
            tasks,
            n,
            k,
            stage,
            max_stack,
            block_i,
            num_warps=1,
        )
        d_prev, d_next = d_next, d_prev

    centers = torch.empty((tiles, k), device=values.device, dtype=torch.float32)
    _dp_centers_kernel[(tiles, k)](
        prefix_0, prefix_1, means, args, centers, n, k, num_warps=1
    )
    if has_weights:
        _dp_sort_centers_kernel[(tiles,)](
            centers, k, triton.next_power_of_2(k), num_warps=1
        )
    return centers


def dp_divide_conquer_local_stack_triton(
    values: torch.Tensor,
    k: int,
    *,
    weights: torch.Tensor | None = None,
    presorted: bool = False,
    candidate_block_size: int = 32,
) -> torch.Tensor:
    """Parallel candidate scans with the D&C task stack stored in CTA state."""
    if candidate_block_size <= 0 or candidate_block_size & (candidate_block_size - 1):
        raise ValueError("candidate_block_size must be a positive power of two")
    shortcut = _dp_center_shortcut(values, k)
    if shortcut is not None:
        return shortcut
    tiles, n = values.shape
    stride = n + 1
    has_weights = weights is not None
    weights = _prepare_dp_weights(values, weights)
    prefix_0, prefix_1, prefix_2, means = _make_dp_prefixes(
        values, weights, presorted
    )

    d_prev = torch.empty((tiles, stride), device=values.device, dtype=torch.float32)
    d_next = torch.empty_like(d_prev)
    _dp_init_kernel[(tiles,)](
        d_prev, n, triton.next_power_of_2(stride), num_warps=4
    )
    args = torch.empty((tiles, k + 1, stride), device=values.device, dtype=torch.int32)
    max_stack = 2 * n.bit_length() + 4
    block_stack = triton.next_power_of_2(max_stack)

    for stage in range(1, k + 1):
        _dp_dc_local_stack_stage_kernel[(tiles,)](
            prefix_0,
            prefix_1,
            prefix_2,
            d_prev,
            d_next,
            args,
            n,
            k,
            stage,
            block_stack,
            candidate_block_size,
            num_warps=1,
        )
        d_prev, d_next = d_next, d_prev

    centers = torch.empty((tiles, k), device=values.device, dtype=torch.float32)
    _dp_centers_kernel[(tiles, k)](
        prefix_0, prefix_1, means, args, centers, n, k, num_warps=1
    )
    if has_weights:
        _dp_sort_centers_kernel[(tiles,)](
            centers, k, triton.next_power_of_2(k), num_warps=1
        )
    return centers


def dp_divide_conquer_fused_stages_triton(
    values: torch.Tensor,
    k: int,
    *,
    weights: torch.Tensor | None = None,
    presorted: bool = False,
    candidate_block_size: int = 32,
) -> torch.Tensor:
    """Run all D&C rows in one stage kernel launch per tile."""
    if candidate_block_size <= 0 or candidate_block_size & (candidate_block_size - 1):
        raise ValueError("candidate_block_size must be a positive power of two")
    shortcut = _dp_center_shortcut(values, k)
    if shortcut is not None:
        return shortcut
    tiles, n = values.shape
    stride = n + 1
    has_weights = weights is not None
    weights = _prepare_dp_weights(values, weights)
    prefix_0, prefix_1, prefix_2, means = _make_dp_prefixes(
        values, weights, presorted
    )

    dp = torch.empty((tiles, 2, stride), device=values.device, dtype=torch.float32)
    args = torch.empty((tiles, k + 1, stride), device=values.device, dtype=torch.int32)
    max_stack = 2 * n.bit_length() + 4
    tasks = torch.empty(
        (tiles, max_stack, 4), device=values.device, dtype=torch.int32
    )
    _dp_dc_fused_stages_kernel[(tiles,)](
        prefix_0,
        prefix_1,
        prefix_2,
        dp,
        args,
        tasks,
        n,
        k,
        max_stack,
        triton.next_power_of_2(stride),
        candidate_block_size,
        num_warps=1,
    )

    centers = torch.empty((tiles, k), device=values.device, dtype=torch.float32)
    _dp_centers_kernel[(tiles, k)](
        prefix_0, prefix_1, means, args, centers, n, k, num_warps=1
    )
    if has_weights:
        _dp_sort_centers_kernel[(tiles,)](
            centers, k, triton.next_power_of_2(k), num_warps=1
        )
    return centers


def dp_smawk_triton(
    values: torch.Tensor,
    k: int,
    *,
    weights: torch.Tensor | None = None,
    presorted: bool = False,
) -> torch.Tensor:
    """Exact 1D k-means DP using SMAWK, O(k n) for sorted inputs.

    Optional nonnegative weights broadcast to ``values`` and are sorted with
    each tile. With ``presorted=True``, values and weights must already be
    aligned/sorted.
    """
    shortcut = _dp_center_shortcut(values, k)
    if shortcut is not None:
        return shortcut
    tiles, n = values.shape
    stride = n + 1
    has_weights = weights is not None
    weights = _prepare_dp_weights(values, weights)
    prefix_0, prefix_1, prefix_2, means = _make_dp_prefixes(
        values, weights, presorted
    )

    d_prev = torch.empty((tiles, stride), device=values.device, dtype=torch.float32)
    d_next = torch.empty_like(d_prev)
    _dp_init_kernel[(tiles,)](
        d_prev, n, triton.next_power_of_2(stride), num_warps=4
    )
    args = torch.empty((tiles, k + 1, stride), device=values.device, dtype=torch.int32)
    max_levels = n.bit_length() + 1
    cols = torch.empty(
        (tiles, max_levels, stride), device=values.device, dtype=torch.int32
    )
    col_counts = torch.empty(
        (tiles, max_levels), device=values.device, dtype=torch.int32
    )

    for stage in range(1, k + 1):
        _dp_smawk_stage_kernel[(tiles,)](
            prefix_0,
            prefix_1,
            prefix_2,
            d_prev,
            d_next,
            args,
            cols,
            col_counts,
            n,
            k,
            stage,
            max_levels,
            num_warps=1,
        )
        d_prev, d_next = d_next, d_prev

    centers = torch.empty((tiles, k), device=values.device, dtype=torch.float32)
    _dp_centers_kernel[(tiles, k)](
        prefix_0, prefix_1, means, args, centers, n, k, num_warps=1
    )
    if has_weights:
        _dp_sort_centers_kernel[(tiles,)](
            centers, k, triton.next_power_of_2(k), num_warps=1
        )
    return centers


def make_emeans_permutations(
    tiles: int, n: int, epochs: int, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    keys = torch.rand(
        (tiles, epochs, n), device=device, generator=generator, dtype=torch.float32
    )
    return torch.argsort(keys, dim=-1).to(torch.int32).contiguous()


def emeans_triton(
    values: torch.Tensor,
    k: int,
    epochs: int = 5,
    *,
    permutations: torch.Tensor | None = None,
    seed: int = 2026,
    batch_law: str = "auto",
):
    tiles, n = values.shape
    law, batch_size, alpha0 = emeans_recipe(
        n, k, epochs, d=1, batch_law=batch_law
    )
    if n % batch_size:
        raise ValueError(
            "this benchmark expects LUT-B n to divide into e-means batches"
        )
    if permutations is None:
        permutations = make_emeans_permutations(
            tiles, n, epochs, values.device, seed
        )

    state_centers = torch.empty((tiles, k), device=values.device, dtype=torch.float32)
    last_labels = torch.empty((tiles, n), device=values.device, dtype=torch.int32)
    steps_per_epoch = n // batch_size
    total_steps = epochs * steps_per_epoch
    _emeans_train_kernel[(tiles,)](
        values,
        permutations,
        state_centers,
        last_labels,
        n,
        k,
        epochs,
        batch_size,
        steps_per_epoch,
        total_steps,
        alpha0,
        num_warps=4,
    )
    centers = torch.empty_like(state_centers)
    _emeans_finalize_kernel[(tiles,)](
        values,
        state_centers,
        last_labels,
        centers,
        n,
        k,
        triton.next_power_of_2(n),
        num_warps=4,
    )
    recipe = {"batch_law": law, "batch_size": batch_size, "alpha0": alpha0}
    return centers, recipe


def normalize_tiles(values: torch.Tensor) -> torch.Tensor:
    tiles, n = values.shape
    normalized = torch.empty_like(values)
    _normalize_tiles_kernel[(tiles,)](
        values, normalized, n, triton.next_power_of_2(n), num_warps=4
    )
    return normalized


def sort_tiles(values: torch.Tensor) -> torch.Tensor:
    tiles, n = values.shape
    sorted_values = torch.empty_like(values)
    _sort_tiles_kernel[(tiles,)](
        values, sorted_values, n, triton.next_power_of_2(n), num_warps=4
    )
    return sorted_values


def reconstruction_mse(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    tiles, n = values.shape
    mse = torch.empty((tiles,), device=values.device, dtype=torch.float32)
    _mse_kernel[(tiles,)](
        values,
        centers,
        mse,
        n,
        centers.shape[1],
        triton.next_power_of_2(n),
        num_warps=4,
    )
    return mse.mean()


def benchmark_ms(fn, warmup_ms: int, rep_ms: int) -> float:
    return float(
        do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
    )


def compare_case(
    n: int,
    tiles: int,
    k: int,
    iterations: int,
    emeans_epochs: int,
    emeans_batch_law: str,
    warmup_ms: int,
    rep_ms: int,
):
    raw = torch.randn((tiles, n), device="cuda", dtype=torch.float32)
    values = normalize_tiles(raw)
    torch.cuda.synchronize()

    lloyd_centers = lloyd_triton(values, k, iterations)
    sorted_values = sort_tiles(values)
    dp_centers = dp_triton(values, k)
    dc_centers = dp_divide_conquer_triton(values, k)
    dc_parallel_centers = dp_divide_conquer_parallel_scan_triton(values, k)
    emeans_centers, emeans_config = emeans_triton(
        values,
        k,
        emeans_epochs,
        seed=2026 + n,
        batch_law=emeans_batch_law,
    )
    lloyd_mse = reconstruction_mse(values, lloyd_centers)
    dp_mse = reconstruction_mse(values, dp_centers)
    dc_mse = reconstruction_mse(values, dc_centers)
    dc_parallel_mse = reconstruction_mse(values, dc_parallel_centers)
    emeans_mse = reconstruction_mse(values, emeans_centers)
    torch.cuda.synchronize()

    emeans_permutations = make_emeans_permutations(
        tiles, n, emeans_epochs, values.device, seed=9876 + n
    )
    lloyd_ms = benchmark_ms(
        lambda: lloyd_triton(values, k, iterations), warmup_ms, rep_ms
    )
    dp_core_ms = benchmark_ms(
        lambda: dp_triton(sorted_values, k, presorted=True), warmup_ms, rep_ms
    )
    dp_end_to_end_ms = benchmark_ms(
        lambda: dp_triton(values, k), warmup_ms, rep_ms
    )
    dc_ms = benchmark_ms(
        lambda: dp_divide_conquer_triton(values, k), warmup_ms, rep_ms
    )
    dc_parallel_ms = benchmark_ms(
        lambda: dp_divide_conquer_parallel_scan_triton(values, k),
        warmup_ms,
        rep_ms,
    )
    emeans_core_ms = benchmark_ms(
        lambda: emeans_triton(
            values,
            k,
            emeans_epochs,
            permutations=emeans_permutations,
            batch_law=emeans_batch_law,
        ),
        warmup_ms,
        rep_ms,
    )
    emeans_end_to_end_ms = benchmark_ms(
        lambda: emeans_triton(
            values,
            k,
            emeans_epochs,
            seed=2026 + n,
            batch_law=emeans_batch_law,
        ),
        warmup_ms,
        rep_ms,
    )
    lloyd_mse_value = lloyd_mse.item()
    dp_mse_value = dp_mse.item()
    dc_mse_value = dc_mse.item()
    dc_parallel_mse_value = dc_parallel_mse.item()
    emeans_mse_value = emeans_mse.item()
    signal_mse_value = values.square().mean().item()
    improvement = (lloyd_mse_value - dp_mse_value) / lloyd_mse_value * 100.0
    print(
        f"N={n:3d} tiles={tiles:3d}: "
        f"Lloyd {lloyd_ms:8.3f} ms ({lloyd_ms * 1000 / tiles:8.2f} us/tile), "
        f"DP core sorted {dp_core_ms:8.3f} ms "
        f"({dp_core_ms * 1000 / tiles:8.2f} us/tile), "
        f"DP incl sort {dp_end_to_end_ms:8.3f} ms "
        f"({dp_end_to_end_ms * 1000 / tiles:8.2f} us/tile), "
        f"DP/Lloyd {dp_core_ms / lloyd_ms:6.2f}x core / "
        f"{dp_end_to_end_ms / lloyd_ms:6.2f}x end-to-end; "
        f"MSE {lloyd_mse_value:.6g} -> {dp_mse_value:.6g} "
        f"({improvement:+.2f}% DP MSE reduction; "
        f"{lloyd_mse_value / signal_mse_value:.2%} -> "
        f"{dp_mse_value / signal_mse_value:.2%} of signal energy)"
    )
    print(
        f"  D&C scalar scan {dc_ms:.3f} ms; parallel candidate scan "
        f"{dc_parallel_ms:.3f} ms ({dc_ms / dc_parallel_ms:.2f}x speedup), "
        f"MSE {dc_mse_value:.6g} -> {dc_parallel_mse_value:.6g}"
    )
    emeans_improvement = (
        (lloyd_mse_value - emeans_mse_value) / lloyd_mse_value * 100.0
    )
    print(
        f"  e-means/{emeans_config['batch_law']} "
        f"B={emeans_config['batch_size']} alpha0={emeans_config['alpha0']:.4f}: "
        f"core {emeans_core_ms:.3f} ms, incl shuffle "
        f"{emeans_end_to_end_ms:.3f} ms; MSE {emeans_mse_value:.6g} "
        f"({emeans_improvement:+.2f}% vs Lloyd; "
        f"{emeans_mse_value / signal_mse_value:.2%} of signal energy)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--emeans-epochs", type=int, default=5)
    parser.add_argument(
        "--emeans-batch-law",
        choices=("auto", "quality", "throughput"),
        default="auto",
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.k & (args.k - 1):
        raise ValueError("--k must be a power of two for the Triton center vector")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    print(f"GPU: {torch.cuda.get_device_name()}; Triton {triton.__version__}")
    print("Inputs are normalized on GPU to the canonical LUT-B E4M3 range.")
    for n in (128, 512):
        compare_case(
            n,
            args.tiles,
            args.k,
            args.iterations,
            args.emeans_epochs,
            args.emeans_batch_law,
            args.warmup_ms,
            args.rep_ms,
        )


if __name__ == "__main__":
    main()
