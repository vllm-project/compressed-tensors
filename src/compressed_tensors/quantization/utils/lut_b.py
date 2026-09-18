# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Canonical logical LUT-B storage and QDQ utilities.

Tiles remain in logical weight order. Runtime backends may swizzle these tensors
into their kernel-specific layouts after checkpoint loading.

The tile geometry (block shape, codebook size and packed byte count) is derived
from the :class:`QuantizationArgs` describing the weight, so a scheme's
``block_structure`` and ``num_bits`` fully determine the on-disk layout. Callers
are expected to have validated the args with :func:`is_lut_b_quantization`
first, so the helpers here read the args directly without re-validating them.
"""

import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)


__all__ = [
    "dequantize_lut_b",
    "fake_quantize_lut_b",
    "is_lut_b_quantization",
    "pack_lut_b_indices",
    "quantize_lut_b",
    "unpack_lut_b_indices",
]


LUT_B_LLOYD_ITERATIONS = 8
LUT_B_MAX_TILES_PER_CHUNK = 4096


def is_lut_b_quantization(args: QuantizationArgs) -> bool:
    """Return whether quantization args describe the canonical LUT-B format."""
    return (
        args.type == QuantizationType.CODEBOOK
        and args.num_bits == 3
        and args.strategy == QuantizationStrategy.BLOCK
        and isinstance(args.block_structure, list)
        and len(args.block_structure) == 2
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


def _snap_to_e4m3(values: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return (
        values.clamp(min=finfo.min, max=finfo.max)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )


def _initialize_centers(values: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    codebook_size = 1 << args.num_bits
    centers = torch.empty(
        values.shape[0],
        codebook_size,
        dtype=torch.float32,
        device=values.device,
    )
    centers[:, 0] = values.mean(dim=1)
    minimum_distance = (values - centers[:, :1]).square()
    for center_index in range(1, codebook_size):
        farthest = minimum_distance.argmax(dim=1, keepdim=True)
        new_center = torch.gather(values, 1, farthest).squeeze(1)
        centers[:, center_index] = new_center
        minimum_distance = torch.minimum(
            minimum_distance,
            (values - new_center[:, None]).square(),
        )
    return _snap_to_e4m3(centers).sort(dim=1).values


def _assign_indices(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    boundaries = ((centers[:, :-1] + centers[:, 1:]) * 0.5).contiguous()
    return torch.searchsorted(boundaries, values.contiguous()).to(torch.int64)


def _fit_codebooks(
    values: torch.Tensor,
    args: QuantizationArgs,
    num_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.to(torch.float32)
    centers = _initialize_centers(values, args)

    for _ in range(num_iterations):
        indices = _assign_indices(values, centers)
        sums = torch.zeros_like(centers)
        counts = torch.zeros_like(centers)
        sums.scatter_add_(1, indices, values)
        counts.scatter_add_(1, indices, torch.ones_like(values))
        updated = sums / counts.clamp_min(1)
        centers = torch.where(counts > 0, updated, centers)
        centers = _snap_to_e4m3(centers).sort(dim=1).values

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
    *,
    max_tiles_per_chunk: int = LUT_B_MAX_TILES_PER_CHUNK,
    num_iterations: int = LUT_B_LLOYD_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create canonical packed indices and E4M3 codebooks for a weight.

    A supplied codebook is used as-is after E4M3 rounding and sorting. This is
    the integration point for calibration-aware codebook creation.
    """
    block_n, block_k = args.block_structure
    codebook_size = 1 << args.num_bits
    packed_tile_bytes = block_n * block_k * args.num_bits // 8

    leading_shape, rows, columns = _validate_weight(weight, args)
    row_tiles = rows // block_n
    column_tiles = columns // block_k
    tile_shape = (*leading_shape, row_tiles, column_tiles)

    if max_tiles_per_chunk <= 0:
        raise ValueError("max_tiles_per_chunk must be positive")
    if num_iterations < 0:
        raise ValueError("num_iterations must be non-negative")
    if codebooks is not None:
        if codebooks.device != weight.device:
            raise ValueError("LUT-B weight and codebooks must be on the same device")
        codebooks = _prepare_codebooks(codebooks, tile_shape, args)

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
        return packed, codebooks

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

    tiles = _weight_to_tiles(weight, args)
    flat_codebooks = (
        codebooks.reshape(-1, codebook_size) if codebooks is not None else None
    )
    flat_packed = packed.reshape(-1, packed_tile_bytes)
    flat_fitted_codebooks = fitted_codebooks.reshape(-1, codebook_size)

    for start in range(0, tiles.shape[0], max_tiles_per_chunk):
        end = min(start + max_tiles_per_chunk, tiles.shape[0])
        values = tiles[start:end].to(torch.float32)
        if flat_codebooks is None:
            indices, centers = _fit_codebooks(values, args, num_iterations)
        else:
            centers = flat_codebooks[start:end]
            indices = _assign_indices(values, centers.to(torch.float32)).to(torch.uint8)
        flat_packed[start:end] = pack_lut_b_indices(indices, args)
        flat_fitted_codebooks[start:end] = centers

    return packed, fitted_codebooks


@torch.no_grad()
def dequantize_lut_b(
    packed: torch.Tensor,
    codebooks: torch.Tensor,
    args: QuantizationArgs,
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
    _validate_codebook_shape(codebooks, tile_shape, args)
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
    values = torch.gather(
        codebooks.reshape(-1, codebook_size).to(dtype),
        1,
        indices.to(torch.int64),
    ).reshape(
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
) -> torch.Tensor:
    """Quantize and dequantize a weight through the canonical LUT-B format."""
    packed, fitted_codebooks = quantize_lut_b(weight, args, codebooks)
    return dequantize_lut_b(packed, fitted_codebooks, args, dtype=weight.dtype)
