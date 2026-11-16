# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Canonical logical LUT-B storage and QDQ utilities.

Tiles remain in logical weight order. Runtime backends may swizzle these tensors
into their kernel-specific layouts after checkpoint loading.

The tile geometry (block shape, codebook size and packed byte count) is derived
from ``block_structure`` and ``num_bits``. The scale tensor geometry is derived
from ``scale_block_structure``; omitting it keeps one scale per codebook tile.
Callers are expected to have validated the args with
:func:`is_lut_b_quantization` first. A codebook tile stores one or more uint8
E8M0 exponents alongside its normalized E4M3 codebook.
"""

import math

import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.lut_b_dp import fit_lut_b_codebooks_dp


__all__ = [
    "dequantize_lut_b",
    "fake_quantize_lut_b",
    "is_lut_b_quantization",
    "pack_lut_b_indices",
    "quantize_lut_b",
    "unpack_lut_b_indices",
]


LUT_B_MAX_TILES_PER_CHUNK = 4096
LUT_B_E8M0_BIAS = 127
LUT_B_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def is_lut_b_quantization(args: QuantizationArgs) -> bool:
    """Return whether quantization args describe the canonical LUT-B format."""
    if args.block_structure is None:
        return False
    scale_block = args.scale_block_structure or args.block_structure
    return (
        args.type == QuantizationType.CODEBOOK
        and args.num_bits == 3
        and args.strategy == QuantizationStrategy.BLOCK
        and isinstance(args.block_structure, list)
        and len(args.block_structure) == 2
        and args.scale_dtype in (None, torch.uint8)
        and isinstance(scale_block, list)
        and len(scale_block) == 2
        and all(
            block_size % scale_size == 0
            for block_size, scale_size in zip(args.block_structure, scale_block)
        )
        and (args.scale_fit_iterations is None or args.scale_fit_iterations >= 0)
    )


def pack_lut_b_indices(indices: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    """Pack eight 3-bit LUT indices into three bytes."""
    if args.num_bits != 3:
        raise ValueError(
            f"LUT-B packing currently supports only num_bits=3, got {args.num_bits}"
        )
    if indices.shape[-1] % 8 != 0:
        raise ValueError("The number of LUT indices must be divisible by 8")
    if indices.device.type != "meta" and torch.any((indices < 0) | (indices > 7)):
        raise ValueError("LUT-B indices must be in the range [0, 7]")

    index_groups = indices.reshape(*indices.shape[:-1], -1, 8).to(torch.int32)
    words = torch.zeros(
        index_groups.shape[:-1],
        dtype=torch.int32,
        device=indices.device,
    )
    for index in range(8):
        words |= index_groups[..., index] << (3 * index)

    return (
        torch.stack(
            (
                words & 0xFF,
                (words >> 8) & 0xFF,
                (words >> 16) & 0xFF,
            ),
            dim=-1,
        )
        .to(torch.uint8)
        .flatten(start_dim=-2)
    )


def unpack_lut_b_indices(packed: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    """Unpack three-byte groups into eight 3-bit LUT indices."""
    if args.num_bits != 3:
        raise ValueError(
            f"LUT-B packing currently supports only num_bits=3, got {args.num_bits}"
        )
    if packed.shape[-1] % 3 != 0:
        raise ValueError("The number of packed bytes must be divisible by 3")

    byte_groups = packed.reshape(*packed.shape[:-1], -1, 3).to(torch.int32)
    words = (
        byte_groups[..., 0] | (byte_groups[..., 1] << 8) | (byte_groups[..., 2] << 16)
    )
    return (
        torch.stack(
            tuple((words >> (3 * index)) & 0x7 for index in range(8)),
            dim=-1,
        )
        .to(torch.uint8)
        .flatten(start_dim=-2)
    )


def _validate_weight(
    weight: torch.Tensor, args: QuantizationArgs
) -> tuple[torch.Size, int, int]:
    if weight.ndim < 2:
        raise ValueError(
            f"LUT-B expects a weight with at least two dimensions, got {weight.shape}"
        )

    block_n, block_k = args.block_structure
    leading_shape = weight.shape[:-2]
    rows, columns = weight.shape[-2:]
    if rows % block_n != 0 or columns % block_k != 0:
        raise ValueError(
            "LUT-B requires the final weight dimensions to be divisible by "
            f"({block_n}, {block_k}), got ({rows}, {columns})"
        )
    return leading_shape, rows, columns


def _weight_to_tiles(weight: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    block_n, block_k = args.block_structure
    leading_shape, rows, columns = _validate_weight(weight, args)
    leading_dims = len(leading_shape)
    return (
        weight.reshape(
            *leading_shape,
            rows // block_n,
            block_n,
            columns // block_k,
            block_k,
        )
        .permute(
            *range(leading_dims),
            leading_dims,
            leading_dims + 2,
            leading_dims + 1,
            leading_dims + 3,
        )
        .reshape(-1, block_n * block_k)
    )


def _importance_to_tiles(
    importance: torch.Tensor | None,
    weight: torch.Tensor,
    args: QuantizationArgs,
    start: int,
    end: int,
) -> torch.Tensor | None:
    """Get importance scores for a flat range of LUT-B tiles."""
    if importance is None:
        return None

    leading_shape, rows, columns = _validate_weight(weight, args)
    block_n, block_k = args.block_structure
    row_tiles = rows // block_n
    column_tiles = columns // block_k
    leading_tiles = math.prod(leading_shape) if leading_shape else 1
    importance = torch.as_tensor(importance, device=weight.device, dtype=torch.float32)
    if importance.ndim == 1 and importance.numel() == columns:
        column_importance = importance.reshape(1, columns).expand(
            leading_tiles, columns
        )
    elif importance.shape == (*leading_shape, columns):
        column_importance = importance.reshape(leading_tiles, columns)
    elif importance.shape == weight.shape:
        if not torch.isfinite(importance).all():
            raise ValueError("LUT-B importance must contain only finite values")
        if torch.any(importance < 0):
            raise ValueError("LUT-B importance must be non-negative")
        return _weight_to_tiles(importance, args)[start:end]
    else:
        raise ValueError(
            "LUT-B importance must have shape [K], [..., K], or match the weight; "
            f"got {tuple(importance.shape)} for weight {tuple(weight.shape)}"
        )

    if not torch.isfinite(column_importance).all():
        raise ValueError("LUT-B importance must contain only finite values")
    if torch.any(column_importance < 0):
        raise ValueError("LUT-B importance must be non-negative")

    tile_ids = torch.arange(start, end, device=weight.device)
    leading_ids = tile_ids // (row_tiles * column_tiles)
    column_tile_ids = tile_ids % column_tiles
    column_ids = column_tile_ids[:, None] * block_k + torch.arange(
        block_k, device=weight.device
    )
    tile_column_importance = column_importance[leading_ids[:, None], column_ids]
    return (
        tile_column_importance.unsqueeze(1)
        .expand(-1, block_n, -1)
        .reshape(end - start, block_n * block_k)
    )


def _scale_lut_b_tiles(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize flattened LUT-B tiles and return their E8M0 scales.

    The exponent places each tile's maximum magnitude in
    ``(E4M3_MAX / 2, E4M3_MAX]``, matching the LUT-B online quantizer. The
    returned scale bytes use the E8M0 bias of 127.
    """
    values = tiles.to(torch.float32)
    amax = values.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax / LUT_B_E4M3_MAX)).clamp(
        -LUT_B_E8M0_BIAS, LUT_B_E8M0_BIAS
    )
    scales = (exponent + LUT_B_E8M0_BIAS).to(torch.uint8)
    normalized = values * torch.exp2(-exponent).unsqueeze(1)
    return normalized, scales


def _lut_b_scale_geometry(
    args: QuantizationArgs,
) -> tuple[int, int, int, int, int, int]:
    """Return tile and scale-block geometry plus scale count per tile."""
    block_n, block_k = args.block_structure
    scale_n, scale_k = args.scale_block_structure or args.block_structure
    if block_n % scale_n or block_k % scale_k:
        raise ValueError(
            "LUT-B scale_block_structure must divide block_structure; got "
            f"{args.scale_block_structure} inside {args.block_structure}"
        )
    scale_rows = block_n // scale_n
    scale_columns = block_k // scale_k
    return (
        block_n,
        block_k,
        scale_n,
        scale_k,
        scale_rows * scale_columns,
        scale_rows,
    )


def _lut_b_scale_grid(args: QuantizationArgs) -> tuple[int, int]:
    block_n, block_k, _, _, _, scale_rows = _lut_b_scale_geometry(args)
    scale_k = (args.scale_block_structure or args.block_structure)[1]
    return scale_rows, block_k // scale_k


def _tiles_to_scale_blocks(
    tiles: torch.Tensor, args: QuantizationArgs
) -> torch.Tensor:
    """View flat tiles as row-major scale blocks: [tiles, blocks, values]."""
    block_n, block_k, scale_n, scale_k, scale_count, _ = _lut_b_scale_geometry(args)
    scale_rows, scale_columns = _lut_b_scale_grid(args)
    return (
        tiles.reshape(-1, block_n, block_k)
        .reshape(-1, scale_rows, scale_n, scale_columns, scale_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, scale_count, scale_n * scale_k)
    )


def _scale_exponents_to_tile(
    exponents: torch.Tensor, args: QuantizationArgs
) -> torch.Tensor:
    """Expand per-scale exponents to one exponent for every value in each tile."""
    block_n, block_k, scale_n, scale_k, scale_count, _ = _lut_b_scale_geometry(args)
    if scale_count == 1:
        return exponents.reshape(-1, 1).expand(-1, block_n * block_k)

    scale_rows, scale_columns = _lut_b_scale_grid(args)
    return (
        exponents.reshape(-1, scale_rows, scale_columns, 1, 1)
        .expand(-1, scale_rows, scale_columns, scale_n, scale_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, block_n * block_k)
    )


def _initial_lut_b_scale_exponents(
    tiles: torch.Tensor, args: QuantizationArgs
) -> torch.Tensor:
    """Initialize E8M0 exponents, using a circular max cutoff for multi-scale."""
    _, _, _, _, scale_count, _ = _lut_b_scale_geometry(args)
    if scale_count == 1:
        _, scales = _scale_lut_b_tiles(tiles)
        return scales.to(torch.int32).sub(LUT_B_E8M0_BIAS).reshape(-1, 1)

    maxima = _tiles_to_scale_blocks(tiles, args).abs().amax(dim=-1)
    valid = maxima > 0
    log_maxima = torch.log2(maxima.clamp_min(torch.finfo(torch.float32).tiny))
    phases = log_maxima - torch.floor(log_maxima)
    _, scale_count = maxima.shape
    sorted_phases = torch.sort(
        torch.where(valid, phases, torch.inf), dim=1
    ).values
    valid_count = valid.sum(dim=1)
    positions = torch.arange(scale_count, device=tiles.device)[None, :]
    safe_count = valid_count.clamp_min(1)[:, None]
    next_position = (positions + 1) % safe_count
    next_phase = sorted_phases.gather(1, next_position)
    gaps = next_phase - sorted_phases
    is_last = positions == (valid_count - 1)[:, None]
    gaps = torch.where(is_last, next_phase + 1.0 - sorted_phases, gaps)
    gaps = torch.where(positions < valid_count[:, None], gaps, -1.0)
    largest_gap = gaps.argmax(dim=1, keepdim=True)
    cut_position = (largest_gap + 1) % safe_count
    cut = sorted_phases.gather(1, cut_position)
    cut = torch.where(valid_count[:, None] > 0, cut, torch.zeros_like(cut))

    exponents = torch.floor(log_maxima - cut)
    exponents = torch.where(valid, exponents, torch.zeros_like(exponents))
    return exponents.clamp(-LUT_B_E8M0_BIAS, LUT_B_E8M0_BIAS).to(torch.int32)


def _lut_b_scale_factors(
    exponents: torch.Tensor, args: QuantizationArgs
) -> torch.Tensor:
    return torch.exp2(_scale_exponents_to_tile(exponents, args).to(torch.float32))


def _normalize_lut_b_tiles(
    tiles: torch.Tensor, exponents: torch.Tensor, args: QuantizationArgs
) -> torch.Tensor:
    _, _, _, _, scale_count, _ = _lut_b_scale_geometry(args)
    if scale_count == 1:
        return tiles * torch.exp2(-exponents.to(torch.float32)).reshape(-1, 1)
    return tiles / _lut_b_scale_factors(exponents, args)


def _lut_b_fit_weights(
    exponents: torch.Tensor,
    args: QuantizationArgs,
    importance: torch.Tensor | None,
) -> torch.Tensor | None:
    """Weight normalized values by scale squared for original-domain DP error."""
    _, _, _, _, scale_count, _ = _lut_b_scale_geometry(args)
    if scale_count == 1 and importance is None:
        return None
    relative_exponents = exponents - exponents.amax(dim=1, keepdim=True)
    squared_scale_weights = torch.exp2(
        _scale_exponents_to_tile(2 * relative_exponents, args).to(torch.float32)
    )
    if importance is not None:
        return importance * squared_scale_weights
    return squared_scale_weights


def _lut_b_tile_errors(
    tiles: torch.Tensor,
    exponents: torch.Tensor,
    centers: torch.Tensor,
    args: QuantizationArgs,
    importance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute original-domain nearest-center squared error per tile."""
    values = _tiles_to_scale_blocks(tiles, args)
    scales = torch.exp2(exponents.to(torch.float32))
    differences = (
        values[..., None]
        - scales[:, :, None, None] * centers[:, None, None, :]
    )
    errors = differences.square().amin(dim=-1)
    if importance is not None:
        errors = errors * _tiles_to_scale_blocks(importance, args)
    return errors.sum(dim=(-1, -2))


def _search_lut_b_scale_exponents(
    tiles: torch.Tensor,
    exponents: torch.Tensor,
    centers: torch.Tensor,
    args: QuantizationArgs,
    radius: int,
    importance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each scale exponent against fixed centers in the original domain."""
    values = _tiles_to_scale_blocks(tiles, args)
    best_exponents = exponents
    centers = centers.to(torch.float32)
    base_scales = torch.exp2(exponents.to(torch.float32))
    base_differences = (
        values[..., None]
        - base_scales[:, :, None, None] * centers[:, None, None, :]
    )
    best_errors = base_differences.square().amin(dim=-1)
    if importance is not None:
        scale_block_importance = _tiles_to_scale_blocks(importance, args)
        scale_block_importance = torch.where(
            scale_block_importance.sum(dim=(1, 2), keepdim=True) > 0,
            scale_block_importance,
            torch.ones_like(scale_block_importance),
        )
        best_errors = best_errors * scale_block_importance
    best_errors = best_errors.sum(dim=-1)
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        candidate = (exponents + offset).clamp(
            -LUT_B_E8M0_BIAS, LUT_B_E8M0_BIAS
        )
        scales = torch.exp2(candidate.to(torch.float32))
        differences = (
            values[..., None]
            - scales[:, :, None, None] * centers[:, None, None, :]
        )
        errors = differences.square().amin(dim=-1)
        if importance is not None:
            errors = errors * scale_block_importance
        errors = errors.sum(dim=-1)
        improved = errors < best_errors
        best_errors = torch.where(improved, errors, best_errors)
        best_exponents = torch.where(improved, candidate, best_exponents)
    return best_exponents, best_errors.sum(dim=1)


def _lut_b_scale_fit_iterations(args: QuantizationArgs) -> int:
    if args.scale_fit_iterations is not None:
        return args.scale_fit_iterations
    _, _, _, _, scale_count, _ = _lut_b_scale_geometry(args)
    return int(scale_count > 1)


def _validate_lut_b_scales(
    scales: torch.Tensor,
    tile_shape: tuple[int, ...],
    device: torch.device,
    args: QuantizationArgs,
) -> None:
    _, _, _, _, scale_count, _ = _lut_b_scale_geometry(args)
    expected_shape = tile_shape if scale_count == 1 else (*tile_shape, scale_count)
    if scales.shape != expected_shape:
        raise ValueError(
            f"LUT-B scale shape must be {expected_shape}, got {scales.shape}"
        )
    if scales.dtype != torch.uint8:
        raise ValueError(f"LUT-B E8M0 scales must use uint8, got {scales.dtype}")
    if scales.device != device:
        raise ValueError("LUT-B weight and scales must be on the same device")
    if scales.device.type != "meta" and torch.any(scales == 0xFF):
        raise ValueError("LUT-B E8M0 scale byte 255 is reserved for NaN")


def _snap_to_e4m3(values: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return (
        values.clamp(min=finfo.min, max=finfo.max)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )


def _assign_indices(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    boundaries = ((centers[:, :-1] + centers[:, 1:]) * 0.5).contiguous()
    return torch.searchsorted(boundaries, values.contiguous()).to(torch.int64)


def _fit_codebooks(
    values: torch.Tensor,
    args: QuantizationArgs,
    importance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit each LUT-B tile with exact 1D k-means DP."""
    values = values.to(torch.float32).contiguous()
    fit_weights = None
    if importance is not None:
        importance = torch.as_tensor(
            importance, device=values.device, dtype=torch.float32
        )
        try:
            importance = torch.broadcast_to(importance, values.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"LUT-B importance shape {tuple(importance.shape)} cannot broadcast "
                f"to tile values {tuple(values.shape)}"
            ) from exc
        if not torch.isfinite(importance).all():
            raise ValueError("LUT-B importance must contain only finite values")
        if torch.any(importance < 0):
            raise ValueError("LUT-B importance must be non-negative")
        # A tile with no observed activation energy still needs a valid codebook.
        importance = torch.where(
            importance.sum(dim=1, keepdim=True) > 0,
            importance,
            torch.ones_like(importance),
        )
        fit_weights = importance / importance.amax(dim=1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        fit_weights = fit_weights.contiguous()

    codebook_size = 1 << args.num_bits
    centers = (
        _snap_to_e4m3(fit_lut_b_codebooks_dp(values, codebook_size, fit_weights))
        .sort(dim=1)
        .values
    )
    indices = _assign_indices(values, centers)
    return indices.to(torch.uint8), centers.to(torch.float8_e4m3fn)


def _validate_codebook_shape(
    codebooks: torch.Tensor,
    tile_shape: tuple[int, ...],
    args: QuantizationArgs,
) -> None:
    expected_shape = (*tile_shape, 1 << args.num_bits)
    if codebooks.shape != expected_shape:
        raise ValueError(
            f"LUT-B codebook shape must be {expected_shape}, got {codebooks.shape}"
        )


def _prepare_codebooks(
    codebooks: torch.Tensor,
    tile_shape: tuple[int, ...],
    args: QuantizationArgs,
) -> torch.Tensor:
    _validate_codebook_shape(codebooks, tile_shape, args)
    if codebooks.device.type == "meta":
        return codebooks.to(torch.float8_e4m3fn)
    return (
        _snap_to_e4m3(codebooks.to(torch.float32))
        .sort(dim=-1)
        .values.to(torch.float8_e4m3fn)
    )


@torch.no_grad()
def quantize_lut_b(
    weight: torch.Tensor,
    args: QuantizationArgs,
    codebooks: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    *,
    importance: torch.Tensor | None = None,
    max_tiles_per_chunk: int = LUT_B_MAX_TILES_PER_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create packed LUT-B indices, E4M3 codebooks and E8M0 scales.

    ``scale_block_structure`` defines the scale granularity inside each
    codebook tile. With one scale per tile, LUT-B fits one DP codebook at the
    usual amax scale. With multiple scales, the default is one provisional DP
    fit, a per-scale exponent search over offsets ±2, and a final DP fit. Set
    ``scale_fit_iterations`` to override that default. Zero performs one DP
    fit without scale search; one performs the provisional fit, a ±2 search,
    and a final fit. Further iterations perform another ±1 search and DP fit.

    Multiple scales are stored on a trailing, row-major scale-block axis; the
    single-scale layout keeps its existing shape.
    Supplied scales are held fixed. Supplied codebooks are used after E4M3
    rounding and sorting, and can score scale candidates when scales are omitted.
    """
    block_n, block_k = args.block_structure
    codebook_size = 1 << args.num_bits
    packed_tile_bytes = block_n * block_k * args.num_bits // 8
    _, _, _, _, scale_count, _ = _lut_b_scale_geometry(args)
    scale_shape_tail = () if scale_count == 1 else (scale_count,)

    leading_shape, rows, columns = _validate_weight(weight, args)
    row_tiles = rows // block_n
    column_tiles = columns // block_k
    tile_shape = (*leading_shape, row_tiles, column_tiles)

    if max_tiles_per_chunk <= 0:
        raise ValueError("max_tiles_per_chunk must be positive")
    if codebooks is not None:
        if codebooks.device != weight.device:
            raise ValueError("LUT-B weight and codebooks must be on the same device")
        codebooks = _prepare_codebooks(codebooks, tile_shape, args)
    if scales is not None:
        _validate_lut_b_scales(scales, tile_shape, weight.device, args)

    if weight.device.type == "meta":
        packed = torch.empty(
            *tile_shape,
            packed_tile_bytes,
            dtype=torch.uint8,
            device=weight.device,
        )
        if codebooks is None:
            codebooks = torch.empty(
                *tile_shape,
                codebook_size,
                dtype=torch.float8_e4m3fn,
                device=weight.device,
            )
        if scales is None:
            scales = torch.empty(
                *tile_shape,
                *scale_shape_tail,
                dtype=torch.uint8,
                device=weight.device,
            )
        return packed, codebooks, scales

    packed = torch.empty(
        *tile_shape,
        packed_tile_bytes,
        dtype=torch.uint8,
        device=weight.device,
    )
    fitted_codebooks = torch.empty(
        *tile_shape,
        codebook_size,
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )
    fitted_scales = (
        scales
        if scales is not None
        else torch.empty(
            *tile_shape,
            *scale_shape_tail,
            dtype=torch.uint8,
            device=weight.device,
        )
    )

    tiles = _weight_to_tiles(weight, args)
    flat_codebooks = (
        codebooks.reshape(-1, codebook_size) if codebooks is not None else None
    )
    flat_packed = packed.reshape(-1, packed_tile_bytes)
    flat_fitted_codebooks = fitted_codebooks.reshape(-1, codebook_size)
    flat_fitted_scales = fitted_scales.reshape(-1, scale_count)
    flat_supplied_scales = (
        scales.reshape(-1, scale_count) if scales is not None else None
    )
    fit_iterations = _lut_b_scale_fit_iterations(args) if scales is None else 0
    chunk_size = (
        min(max_tiles_per_chunk, 256) if fit_iterations else max_tiles_per_chunk
    )

    for start in range(0, tiles.shape[0], chunk_size):
        end = min(start + chunk_size, tiles.shape[0])
        raw_values = tiles[start:end].to(torch.float32)
        if flat_supplied_scales is None:
            exponents = _initial_lut_b_scale_exponents(raw_values, args)
        else:
            exponents = flat_supplied_scales[start:end].to(torch.int32)
            exponents = exponents - LUT_B_E8M0_BIAS

        values = _normalize_lut_b_tiles(raw_values, exponents, args)
        chunk_importance = _importance_to_tiles(
            importance, weight, args, start, end
        )
        if flat_codebooks is None:
            fit_weights = _lut_b_fit_weights(
                exponents, args, chunk_importance
            )
            _, centers = _fit_codebooks(values, args, importance=fit_weights)
            centers = centers.to(torch.float32)
        else:
            centers = flat_codebooks[start:end].to(torch.float32)

        for iteration in range(fit_iterations):
            radius = 2 if iteration == 0 else 1
            exponents, fixed_center_errors = _search_lut_b_scale_exponents(
                raw_values,
                exponents,
                centers,
                args,
                radius,
                importance=chunk_importance,
            )
            if flat_codebooks is None:
                values = _normalize_lut_b_tiles(raw_values, exponents, args)
                fit_weights = _lut_b_fit_weights(
                    exponents, args, chunk_importance
                )
                _, proposed_centers = _fit_codebooks(
                    values, args, importance=fit_weights
                )
                proposed_centers = proposed_centers.to(torch.float32)
                proposed_errors = _lut_b_tile_errors(
                    raw_values,
                    exponents,
                    proposed_centers,
                    args,
                    importance=chunk_importance,
                )
                improved = proposed_errors <= fixed_center_errors
                centers = torch.where(
                    improved[:, None], proposed_centers, centers
                )

        if flat_supplied_scales is None:
            flat_fitted_scales[start:end] = (
                exponents + LUT_B_E8M0_BIAS
            ).to(torch.uint8)

        values = _normalize_lut_b_tiles(raw_values, exponents, args)
        indices = _assign_indices(values, centers.to(torch.float32)).to(torch.uint8)
        flat_packed[start:end] = pack_lut_b_indices(indices, args)
        flat_fitted_codebooks[start:end] = centers

    return packed, fitted_codebooks, fitted_scales


@torch.no_grad()
def dequantize_lut_b(
    packed: torch.Tensor,
    codebooks: torch.Tensor,
    args: QuantizationArgs,
    scales: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct a logical ``[..., N, K]`` weight from LUT-B tensors."""
    block_n, block_k = args.block_structure
    codebook_size = 1 << args.num_bits
    packed_tile_bytes = block_n * block_k * args.num_bits // 8

    if packed.ndim < 3 or packed.shape[-1] != packed_tile_bytes:
        raise ValueError(f"Unexpected packed LUT-B shape {packed.shape}")
    if packed.dtype != torch.uint8:
        raise ValueError(f"Canonical LUT-B indices must use uint8, got {packed.dtype}")
    if packed.device != codebooks.device:
        raise ValueError(
            "LUT-B packed indices and codebooks must be on the same device"
        )

    leading_shape = packed.shape[:-3]
    row_tiles, column_tiles = packed.shape[-3:-1]
    tile_shape = (*leading_shape, row_tiles, column_tiles)
    _, _, _, _, scale_count, _ = _lut_b_scale_geometry(args)
    scale_shape_tail = () if scale_count == 1 else (scale_count,)
    _validate_codebook_shape(codebooks, tile_shape, args)
    if scales is None:
        # Compatibility for callers with unscaled codebooks from the earlier
        # LUT-B representation.
        scales = torch.full(
            (*tile_shape, *scale_shape_tail),
            LUT_B_E8M0_BIAS,
            dtype=torch.uint8,
            device=packed.device,
        )
    _validate_lut_b_scales(scales, tile_shape, packed.device, args)
    if codebooks.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "Canonical LUT-B codebooks must use torch.float8_e4m3fn, "
            f"got {codebooks.dtype}"
        )
    if packed.device.type == "meta":
        return torch.empty(
            *leading_shape,
            row_tiles * block_n,
            column_tiles * block_k,
            dtype=dtype,
            device=packed.device,
        )

    indices = unpack_lut_b_indices(packed, args).reshape(-1, block_n * block_k)
    tile_codebooks = codebooks.reshape(-1, codebook_size).to(torch.float32)
    exponents = scales.reshape(-1, scale_count).to(torch.int32) - LUT_B_E8M0_BIAS
    if scale_count == 1:
        scales_per_tile = torch.exp2(exponents[:, 0].to(torch.float32))
        scaled_codebooks = (tile_codebooks * scales_per_tile[:, None]).to(dtype)
        values = torch.gather(scaled_codebooks, 1, indices.to(torch.int64))
    else:
        values = torch.gather(
            tile_codebooks,
            1,
            indices.to(torch.int64),
        )
        values = (
            values * _lut_b_scale_factors(exponents, args)
        ).to(dtype)
    values = values.reshape(
        *leading_shape,
        row_tiles,
        column_tiles,
        block_n,
        block_k,
    )
    leading_dims = len(leading_shape)
    return values.permute(
        *range(leading_dims),
        leading_dims,
        leading_dims + 2,
        leading_dims + 1,
        leading_dims + 3,
    ).reshape(
        *leading_shape,
        row_tiles * block_n,
        column_tiles * block_k,
    )


@torch.no_grad()
def fake_quantize_lut_b(
    weight: torch.Tensor,
    args: QuantizationArgs,
    codebooks: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    *,
    importance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize and dequantize a weight through the canonical LUT-B format."""
    packed, fitted_codebooks, fitted_scales = quantize_lut_b(
        weight, args, codebooks, scales, importance=importance
    )
    return dequantize_lut_b(
        packed, fitted_codebooks, args, fitted_scales, dtype=weight.dtype
    )
