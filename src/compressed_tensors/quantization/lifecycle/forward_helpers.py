# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil

import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    round_to_quantized_type_args,
)
from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant
from compressed_tensors.quantization.utils.fp4_utils import _round_to_fp4
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import tl, triton, triton_req


# Quantization type constants for Triton kernel
QUANT_TYPE_INT = tl.constexpr(0)
QUANT_TYPE_FLOAT = tl.constexpr(1)


@triton.jit
def _quantize_dequantize_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    global_scale_ptr: tl.tensor,
    q_min_ptr: tl.tensor,
    q_max_ptr: tl.tensor,
    # Note: unused strides for tensors with fewer dimensions are set to 0.
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    dim_0,
    dim_1,
    dim_2,
    dim_3,
    group_size,
    num_scale_cols,
    quant_type: tl.constexpr,  # QUANT_TYPE_INT or QUANT_TYPE_FLOAT
    num_bits: tl.constexpr,  # 4 or 8
    use_intel_libdevice: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    has_zero_point: tl.constexpr,
    has_global_scale: tl.constexpr,
):
    """
    Fused quantize-dequantize kernel for per-group/channel scales.
    Performs: output = ((clamp(round(x / scale + zp)) - zp) * scale

    Handles tensors up to 4D by treating them as a 2D view:
    - row indices span dim_0 * dim_1
    - col indices span dim_2 * dim_3

    Scale is expected to be contiguous and indexed linearly as:
    scale_offset = (idx_0 * dim_1 + idx_1) * num_scale_cols + tile_c // group_size
    """
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    tile_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    tile_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    idx_0 = tile_r // dim_1
    idx_1 = tile_r % dim_1

    idx_2 = tile_c // dim_3
    idx_3 = tile_c % dim_3

    # Compute 2D offset matrices via broadcasting:
    # - [:, None] reshapes [R] -> [R, 1] (column vector for row indices)
    # - [None, :] reshapes [C] -> [1, C] (row vector for col indices)
    # Adding them produces [R, C] matrix of all (row, col) offset combinations
    input_offsets = (
        idx_0[:, None] * input_stride_0
        + idx_1[:, None] * input_stride_1
        + idx_2[None, :] * input_stride_2
        + idx_3[None, :] * input_stride_3
    )

    output_offsets = (
        idx_0[:, None] * output_stride_0
        + idx_1[:, None] * output_stride_1
        + idx_2[None, :] * output_stride_2
        + idx_3[None, :] * output_stride_3
    )

    scale_row_idx = idx_0 * dim_1 + idx_1
    scale_col_idx = tile_c // group_size
    scale_offsets = scale_row_idx[:, None] * num_scale_cols + scale_col_idx[None, :]

    masks_0 = idx_0 < dim_0
    masks_1 = idx_1 < dim_1
    masks_2 = idx_2 < dim_2
    masks_3 = idx_3 < dim_3
    masks = masks_0[:, None] & masks_1[:, None] & masks_2[None, :] & masks_3[None, :]

    num_scale_elements = dim_0 * dim_1 * num_scale_cols
    scale_masks = scale_offsets < num_scale_elements

    # Load inputs
    x = tl.load(input_ptr + input_offsets, masks, 0.0)
    scale = tl.load(scale_ptr + scale_offsets, scale_masks, 1.0)
    q_min = tl.load(q_min_ptr)
    q_max = tl.load(q_max_ptr)

    # Apply global scale if present
    if has_global_scale:
        global_scale = tl.load(global_scale_ptr)
        scale = tl.div_rn(scale.to(tl.float32), global_scale.to(tl.float32))

    # Quantize: x / scale + zero_point (using round-to-nearest division)
    quantized = tl.div_rn(x.to(tl.float32), scale.to(tl.float32))

    if has_zero_point:
        zero_point = tl.load(zero_point_ptr + scale_offsets, scale_masks, 0.0)
        quantized = quantized + zero_point

    # Clamp and round based on quantization type
    if quant_type == QUANT_TYPE_INT:
        quantized = tl.clamp(quantized, q_min, q_max)
        if use_intel_libdevice:
            quantized = tl.extra.intel.libdevice.rint(quantized)
        else:
            quantized = tl.extra.cuda.libdevice.rint(quantized)
    else:  # QUANT_TYPE_FLOAT
        quantized = tl.clamp(quantized, q_min, q_max)
        if num_bits == 4:
            quantized = _round_to_fp4(quantized)
        elif num_bits == 8:
            quantized = quantized.to(tl.float8e4nv).to(quantized.dtype)

    # Dequantize: (quantized - zero_point) * scale
    # Note: for FP4, quantized is bfloat16 but scale is float32,
    # so the result will be promoted to float32
    if has_zero_point:
        output = (quantized - zero_point) * scale
    else:
        output = quantized * scale

    tl.store(output_ptr + output_offsets, output, masks)


def _needs_fp8(*tensors, args: QuantizationArgs) -> bool:
    """Check if operation involves FP8 (dtype or quantization type)."""
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    has_fp8_tensor = any(t is not None and t.dtype in fp8_dtypes for t in tensors)
    is_fp8_quant = args.type == QuantizationType.FLOAT and args.num_bits == 8
    return has_fp8_tensor or is_fp8_quant


def _is_fp8_supported(device: torch.device) -> bool:
    """Check if device supports FP8 natively."""
    if device.type == "cuda":
        major, _ = torch.get_device_module().get_device_capability(device)
        return major >= 9  # SM90+ (Hopper/Ada)
    elif device.type == "xpu":
        # Intel XPU: Triton FP8 casting works on the current backend.
        return True
    return False


def _apply_quantize_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: torch.dtype | None,
    do_quantize: bool,
    do_dequantize: bool,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Dispatch to the appropriate quantization kernel."""
    if do_quantize and do_dequantize:
        # Fusion only benefits Triton - use fused path only if Triton is available
        if _quantize_dequantize_triton_req(
            x, scale, zero_point, q_min, q_max, args, global_scale
        ):
            return _quantize_dequantize_triton(
                x=x,
                scale=scale,
                zero_point=zero_point,
                q_min=q_min,
                q_max=q_max,
                args=args,
                global_scale=global_scale,
            )
        else:
            # PyTorch fallback: just call quantize then dequantize (no fusion benefit)
            quantized = _quantize(
                x=x,
                scale=scale,
                zero_point=zero_point,
                q_min=q_min,
                q_max=q_max,
                args=args,
                dtype=None,  # Keep in float for dequantize
                global_scale=global_scale,
            )
            return _dequantize(
                x_q=quantized,
                scale=scale,
                zero_point=zero_point,
                dtype=x.dtype,
                global_scale=global_scale,
                args=args,
            )
    elif do_quantize:
        return _quantize(
            x=x,
            scale=scale,
            zero_point=zero_point,
            q_min=q_min,
            q_max=q_max,
            args=args,
            dtype=dtype,
            global_scale=global_scale,
        )
    else:
        return _dequantize(
            x_q=x,
            scale=scale,
            zero_point=zero_point,
            global_scale=global_scale,
            args=args,
        )


def _process_block(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    dtype: torch.dtype | None,
    do_quantize: bool,
    do_dequantize: bool,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Blockwise quantization: pad, reshape into 2D blocks, quantize, restore."""
    original_shape = x.shape
    block_height, block_width = args.block_structure

    x = maybe_pad_tensor_for_block_quant(x, args.block_structure)
    padded_shape = x.shape

    # reshape into blocks and transpose to make each block contiguous
    num_rows_blocks = padded_shape[0] // block_height
    num_cols_blocks = padded_shape[1] // block_width
    x_blocks = x.reshape(
        num_rows_blocks,
        block_height,
        num_cols_blocks,
        block_width,
    ).transpose(1, 2)

    # expand scale/zero_point for block broadcasting
    sb = scale.unsqueeze(-1).unsqueeze(-1)
    zb = zero_point.unsqueeze(-1).unsqueeze(-1) if zero_point is not None else None

    x_blocks = _apply_quantize_op(
        x_blocks,
        sb,
        zb,
        q_min,
        q_max,
        args,
        dtype,
        do_quantize,
        do_dequantize,
        global_scale,
    )

    # restore padded shape
    output = x_blocks.transpose(1, 2).reshape(padded_shape)

    # truncate to original dimensions if padding was applied
    if original_shape != padded_shape:
        output = output[tuple([slice(v) for v in original_shape])]

    return output


def _process_group(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    args: QuantizationArgs,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    dtype: torch.dtype | None,
    do_quantize: bool,
    do_dequantize: bool,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Group/tensor-group quantization: reshape into groups, quantize, restore."""
    group_size = args.group_size
    output_dtype = dtype if dtype is not None else x.dtype
    columns = x.shape[-1]

    while scale.ndim < 2:
        scale = scale.unsqueeze(1)
        zero_point = zero_point.unsqueeze(1) if zero_point is not None else None

    if columns >= group_size and columns % group_size != 0:
        raise ValueError(
            "tensor column shape must be divisble "
            f"by the given group_size {group_size} but got {columns}"
        )

    # reshape last dim into (num_groups, group_size)
    reshaped_dims = (ceil(x.shape[-1] / group_size), group_size)
    x = x.unflatten(-1, reshaped_dims)

    output = _apply_quantize_op(
        x,
        scale.unsqueeze(-1),
        zero_point.unsqueeze(-1) if zero_point is not None else None,
        q_min,
        q_max,
        args,
        dtype,
        do_quantize,
        do_dequantize,
        global_scale,
    )

    output = output.flatten(start_dim=-2).to(output_dtype)

    return output


def _quantize_dequantize_triton_req(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    global_scale: torch.Tensor | None = None,
) -> bool:
    return triton_req(x) and (
        not _needs_fp8(x, scale, zero_point, global_scale, args=args)
        or _is_fp8_supported(x.device)
    )


@torch.no_grad()
def _quantize_dequantize_triton(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton implementation of fused quantize-then-dequantize."""
    num_rows = x.shape[0]
    scale, zero_point = adapt_scale_and_zp_for_triton(scale, zero_point, num_rows)

    original_shape = x.shape

    quant_type = (
        QUANT_TYPE_INT if args.type == QuantizationType.INT.value else QUANT_TYPE_FLOAT
    )
    num_bits = args.num_bits

    # Determine dimensions based on strategy (mirroring _quantize_triton logic)
    if args.strategy == QuantizationStrategy.BLOCK:
        dim_0, dim_1, dim_2, dim_3 = x.shape
        group_size = dim_2 * dim_3  # all col elements share same scale
        num_scale_cols = 1  # one scale per (idx_0, idx_1) pair
    elif args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        dim_0 = 1
        dim_1, dim_2, dim_3 = x.shape
        group_size = dim_3
        num_scale_cols = dim_2  # num_groups
    elif args.strategy in (QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL):
        dim_0 = 1
        dim_1, dim_3 = x.shape
        dim_2 = 1
        group_size = dim_3  # all cols share same scale
        num_scale_cols = 1  # one scale per row
    else:
        raise ValueError(f"Unsupported quantization strategy: {args.strategy}")

    num_rows = dim_0 * dim_1
    num_cols = dim_2 * dim_3
    block_size_r: int = 32
    block_size_c: int = 32

    def grid(META):
        return (
            triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
            triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
        )

    output = torch.empty_like(x)

    x_strides = x.stride()
    out_strides = output.stride()

    if args.strategy == QuantizationStrategy.BLOCK:
        input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x_strides
        (
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
        ) = out_strides
    elif args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        input_stride_0 = 0
        input_stride_1, input_stride_2, input_stride_3 = x_strides
        output_stride_0 = 0
        output_stride_1, output_stride_2, output_stride_3 = out_strides
    else:
        input_stride_0 = 0
        input_stride_1, input_stride_3 = x_strides
        input_stride_2 = 0
        output_stride_0 = 0
        output_stride_1, output_stride_3 = out_strides
        output_stride_2 = 0

    with torch.get_device_module().device(x.device):
        _quantize_dequantize_kernel[grid](
            output,
            x,
            scale,
            zero_point if zero_point is not None else x,  # dummy pointer
            global_scale if global_scale is not None else x,  # dummy pointer
            q_min,
            q_max,
            input_stride_0,
            input_stride_1,
            input_stride_2,
            input_stride_3,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
            dim_0,
            dim_1,
            dim_2,
            dim_3,
            group_size,
            num_scale_cols,
            quant_type=quant_type,
            num_bits=num_bits,
            use_intel_libdevice=x.device.type == "xpu",
            BLOCK_SIZE_R=block_size_r,
            BLOCK_SIZE_C=block_size_c,
            has_zero_point=zero_point is not None,
            has_global_scale=global_scale is not None,
        )

    return output.reshape(original_shape)


# Alias for tests/backward compatibility
def _quantize_dequantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Helper function for fused quantize-dequantize.
    This is just a convenience wrapper around _apply_quantize_op.
    """
    return _apply_quantize_op(
        x=x,
        scale=scale,
        zero_point=zero_point,
        q_min=q_min,
        q_max=q_max,
        args=args,
        dtype=None,
        do_quantize=True,
        do_dequantize=True,
        global_scale=global_scale,
    )


@triton.jit
def _dequantize_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    global_scale_ptr: tl.tensor,
    # Note: unused strides for tensors with fewer dimensions are set to 0.
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    dim_0,
    dim_1,
    dim_2,
    dim_3,
    group_size,
    num_scale_cols,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    has_zero_point: tl.constexpr,
    has_global_scale: tl.constexpr,
):
    """General dequantize kernel using explicit strides.

    Handles tensors up to 4D by treating them as a 2D view:
    - row indices span dim_0 * dim_1
    - col indices span dim_2 * dim_3

    Scale is expected to be contiguous and indexed linearly as:
    scale_offset = (idx_0 * dim_1 + idx_1) * num_scale_cols + tile_c // group_size
    """
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    tile_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    tile_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    idx_0 = tile_r // dim_1
    idx_1 = tile_r % dim_1

    idx_2 = tile_c // dim_3
    idx_3 = tile_c % dim_3

    # Compute 2D offset matrices via broadcasting:
    # - [:, None] reshapes [R] -> [R, 1] (column vector for row indices)
    # - [None, :] reshapes [C] -> [1, C] (row vector for col indices)
    # Adding them produces [R, C] matrix of all (row, col) offset combinations
    input_offsets = (
        idx_0[:, None] * input_stride_0
        + idx_1[:, None] * input_stride_1
        + idx_2[None, :] * input_stride_2
        + idx_3[None, :] * input_stride_3
    )

    output_offsets = (
        idx_0[:, None] * output_stride_0
        + idx_1[:, None] * output_stride_1
        + idx_2[None, :] * output_stride_2
        + idx_3[None, :] * output_stride_3
    )

    scale_row_idx = idx_0 * dim_1 + idx_1
    scale_col_idx = tile_c // group_size
    scale_offsets = scale_row_idx[:, None] * num_scale_cols + scale_col_idx[None, :]

    masks_0 = idx_0 < dim_0
    masks_1 = idx_1 < dim_1
    masks_2 = idx_2 < dim_2
    masks_3 = idx_3 < dim_3
    masks = masks_0[:, None] & masks_1[:, None] & masks_2[None, :] & masks_3[None, :]

    num_scale_elements = dim_0 * dim_1 * num_scale_cols
    scale_masks = scale_offsets < num_scale_elements

    input = tl.load(input_ptr + input_offsets, masks, 0.0)
    scale = tl.load(scale_ptr + scale_offsets, scale_masks, 1.0)

    if has_global_scale:
        global_scale = tl.load(global_scale_ptr)
        scale = scale / global_scale.to(scale.dtype)

    # Dequantize: (x_q - zero_point) * scale
    if has_zero_point:
        zero_point = tl.load(zero_point_ptr + scale_offsets, scale_masks, 0.0)
        output = (input - zero_point) * scale
    else:
        output = input * scale

    tl.store(output_ptr + output_offsets, output, masks)


# Quantization type constants for Triton kernel
QUANT_TYPE_INT = tl.constexpr(0)
QUANT_TYPE_FLOAT = tl.constexpr(1)


@triton.jit
def _quantize_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    q_min_ptr: tl.tensor,
    q_max_ptr: tl.tensor,
    global_scale_ptr: tl.tensor,
    # Note: unused strides for tensors with fewer dimensions are set to 0.
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    dim_0,
    dim_1,
    dim_2,
    dim_3,
    group_size,
    num_scale_cols,
    quant_type: tl.constexpr,  # QUANT_TYPE_INT or QUANT_TYPE_FLOAT
    num_bits: tl.constexpr,  # 4 or 8
    use_intel_libdevice: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    has_zero_point: tl.constexpr,
    has_global_scale: tl.constexpr,
):
    """General quantize kernel using explicit strides.

    Handles tensors up to 4D by treating them as a 2D view:
    - row indices span dim_0 * dim_1
    - col indices span dim_2 * dim_3

    Scale is expected to be contiguous and indexed linearly as:
    scale_offset = (idx_0 * dim_1 + idx_1) * num_scale_cols + tile_c // group_size
    """
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    tile_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    tile_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    idx_0 = tile_r // dim_1
    idx_1 = tile_r % dim_1

    idx_2 = tile_c // dim_3
    idx_3 = tile_c % dim_3

    # Compute 2D offset matrices via broadcasting:
    # - [:, None] reshapes [R] -> [R, 1] (column vector for row indices)
    # - [None, :] reshapes [C] -> [1, C] (row vector for col indices)
    # Adding them produces [R, C] matrix of all (row, col) offset combinations
    input_offsets = (
        idx_0[:, None] * input_stride_0
        + idx_1[:, None] * input_stride_1
        + idx_2[None, :] * input_stride_2
        + idx_3[None, :] * input_stride_3
    )

    output_offsets = (
        idx_0[:, None] * output_stride_0
        + idx_1[:, None] * output_stride_1
        + idx_2[None, :] * output_stride_2
        + idx_3[None, :] * output_stride_3
    )

    scale_row_idx = idx_0 * dim_1 + idx_1
    scale_col_idx = tile_c // group_size
    scale_offsets = scale_row_idx[:, None] * num_scale_cols + scale_col_idx[None, :]

    masks_0 = idx_0 < dim_0
    masks_1 = idx_1 < dim_1
    masks_2 = idx_2 < dim_2
    masks_3 = idx_3 < dim_3
    masks = masks_0[:, None] & masks_1[:, None] & masks_2[None, :] & masks_3[None, :]

    num_scale_elements = dim_0 * dim_1 * num_scale_cols
    scale_masks = scale_offsets < num_scale_elements

    input = tl.load(input_ptr + input_offsets, masks, 0.0)
    scale = tl.load(scale_ptr + scale_offsets, scale_masks, 1.0)

    if has_global_scale:
        global_scale = tl.load(global_scale_ptr)
        scale = tl.div_rn(scale.to(tl.float32), global_scale.to(tl.float32))

    output = tl.div_rn(input.to(tl.float32), scale.to(tl.float32))

    if has_zero_point:
        zero_point = tl.load(zero_point_ptr + scale_offsets, scale_masks, 0.0)
        output += zero_point

    # clamp and round (equivalent to round_to_quantized_type_args)
    q_min = tl.load(q_min_ptr)
    q_max = tl.load(q_max_ptr)
    if quant_type == QUANT_TYPE_INT:
        output = tl.clamp(output, q_min, q_max)
        if use_intel_libdevice:
            output = tl.extra.intel.libdevice.rint(output)
        else:
            output = tl.extra.cuda.libdevice.rint(output)
    elif quant_type == QUANT_TYPE_FLOAT:
        output = tl.clamp(output, q_min, q_max)
        if num_bits == 4:
            output = _round_to_fp4(output)
        elif num_bits == 8:
            output = output.to(tl.float8e4nv).to(output.dtype)

    tl.store(output_ptr + output_offsets, output, masks)


def adapt_scale_and_zp_for_triton(
    scale: torch.Tensor, zero_point: torch.Tensor | None, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Adapt scale and zero point for Triton kernel.
    This is required when we use group strategies, so that Triton
    can read the correct scale and zero point for each group.

    Note: We keep scale/zp contiguous because they are small tensors
    (one value per row/group), so contiguous() is cheap
    """
    if scale.ndim == 0:
        scale = scale.expand(num_rows, 1)
    elif scale.ndim == 1:
        scale = scale.unsqueeze(1).expand(num_rows, 1)
    elif scale.shape[0] == 1:
        scale = scale.expand(num_rows, -1)
    scale = scale.contiguous()

    if zero_point is not None:
        if zero_point.ndim == 0:
            zero_point = zero_point.expand(num_rows, 1)
        elif zero_point.ndim == 1:
            zero_point = zero_point.unsqueeze(1).expand(num_rows, 1)
        elif zero_point.shape[0] == 1:
            zero_point = zero_point.expand(num_rows, -1)
        zero_point = zero_point.contiguous()
    return scale, zero_point


def _quantize_triton_req(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
) -> bool:
    return triton_req(x) and (
        not _needs_fp8(x, scale, zero_point, global_scale, args=args)
        or _is_fp8_supported(x.device)
    )


@torch.no_grad()
@ImplBackend.register("_quantize", _quantize_triton_req, 0)
def _quantize_triton(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    num_rows = x.shape[0]
    scale, zero_point = adapt_scale_and_zp_for_triton(scale, zero_point, num_rows)

    original_shape = x.shape

    quant_type = (
        QUANT_TYPE_INT if args.type == QuantizationType.INT else QUANT_TYPE_FLOAT
    )
    num_bits = args.num_bits

    if args.strategy == QuantizationStrategy.BLOCK:
        dim_0, dim_1, dim_2, dim_3 = x.shape
        group_size = dim_2 * dim_3  # all col elements share same scale
        num_scale_cols = 1  # one scale per (idx_0, idx_1) pair
    elif args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        dim_0 = 1
        dim_1, dim_2, dim_3 = x.shape
        group_size = dim_3
        num_scale_cols = dim_2  # num_groups
    elif args.strategy in (QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL):
        dim_0 = 1
        dim_1, dim_3 = x.shape
        dim_2 = 1
        group_size = dim_3  # all cols share same scale
        num_scale_cols = 1  # one scale per row
    else:
        raise ValueError(f"Unsupported quantization strategy: {args.strategy}")

    num_rows = dim_0 * dim_1
    num_cols = dim_2 * dim_3
    block_size_r: int = 32
    block_size_c: int = 32

    def grid(META):
        return (
            triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
            triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
        )

    quantized_value = torch.empty_like(x)

    x_strides = x.stride()
    out_strides = quantized_value.stride()

    if args.strategy == QuantizationStrategy.BLOCK:
        input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x_strides
        (
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
        ) = out_strides
    elif args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        input_stride_0 = 0
        input_stride_1, input_stride_2, input_stride_3 = x_strides
        output_stride_0 = 0
        output_stride_1, output_stride_2, output_stride_3 = out_strides
    else:
        input_stride_0 = 0
        input_stride_1, input_stride_3 = x_strides
        input_stride_2 = 0
        output_stride_0 = 0
        output_stride_1, output_stride_3 = out_strides
        output_stride_2 = 0

    output_dtype = dtype if dtype is not None else x.dtype

    with torch.get_device_module().device(x.device):
        _quantize_kernel[grid](
            quantized_value,
            x,
            scale,
            zero_point if zero_point is not None else x,  # pass x as dummy
            q_min,
            q_max,
            global_scale if global_scale is not None else x,  # pass x as dummy
            input_stride_0,
            input_stride_1,
            input_stride_2,
            input_stride_3,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
            dim_0,
            dim_1,
            dim_2,
            dim_3,
            group_size,
            num_scale_cols,
            quant_type=quant_type,
            num_bits=num_bits,
            use_intel_libdevice=x.device.type == "xpu",
            BLOCK_SIZE_R=block_size_r,
            BLOCK_SIZE_C=block_size_c,
            has_zero_point=zero_point is not None,
            has_global_scale=global_scale is not None,
        )

    quantized_value = quantized_value.reshape(original_shape)

    return quantized_value.to(output_dtype)


@torch.no_grad()
@ImplBackend.entrypoint("_quantize")
def _quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if global_scale is not None:
        scale = scale / global_scale

    scaled = x / scale
    if zero_point is not None:
        scaled += zero_point.to(x.dtype)
    quantized_ground = round_to_quantized_type_args(
        tensor=scaled, args=args, min=q_min, max=q_max
    )
    if dtype is not None:
        quantized_ground = quantized_ground.to(dtype)
    return quantized_ground


def _dequantize_triton_req(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
    args: QuantizationArgs | None = None,
) -> bool:
    return triton_req(x_q)


@torch.no_grad()
@ImplBackend.register("_dequantize", _dequantize_triton_req, 0)
def _dequantize_triton(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
    args: QuantizationArgs | None = None,
) -> torch.Tensor:
    """Triton implementation of dequantization."""
    original_shape = x_q.shape

    # Convert to float for computation
    x_q = x_q.to(scale.dtype)

    # Adapt scale and zero_point for Triton kernel
    num_rows = x_q.shape[0]
    scale, zero_point = adapt_scale_and_zp_for_triton(scale, zero_point, num_rows)

    # Determine dimensions based on strategy (mirroring _quantize logic)
    if args is not None and args.strategy == QuantizationStrategy.BLOCK:
        dim_0, dim_1, dim_2, dim_3 = x_q.shape
        group_size = dim_2 * dim_3  # all col elements share same scale
        num_scale_cols = 1  # one scale per (idx_0, idx_1) pair
    elif args is not None and args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        dim_0 = 1
        dim_1, dim_2, dim_3 = x_q.shape
        group_size = dim_3
        num_scale_cols = dim_2  # num_groups
    elif args is not None and args.strategy in (
        QuantizationStrategy.TENSOR,
        QuantizationStrategy.CHANNEL,
    ):
        dim_0 = 1
        dim_1, dim_3 = x_q.shape
        dim_2 = 1
        group_size = dim_3  # all cols share same scale
        num_scale_cols = 1  # one scale per row
    else:
        # Fallback: infer from tensor shape (legacy behavior)
        if x_q.ndim == 4:
            dim_0, dim_1, dim_2, dim_3 = x_q.shape
            group_size = dim_2 * dim_3
            num_scale_cols = 1
        elif x_q.ndim == 3:
            dim_0 = 1
            dim_1, dim_2, dim_3 = x_q.shape
            group_size = dim_3
            num_scale_cols = dim_2
        elif x_q.ndim == 2:
            dim_0 = 1
            dim_1, dim_3 = x_q.shape
            dim_2 = 1
            group_size = dim_3
            num_scale_cols = 1
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {x_q.ndim}D")

    num_rows = dim_0 * dim_1
    num_cols = dim_2 * dim_3
    block_size_r: int = 32
    block_size_c: int = 32

    def grid(META):
        return (
            triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
            triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
        )

    dequant_value = torch.empty_like(x_q)

    x_strides = x_q.stride()
    out_strides = dequant_value.stride()

    # Compute strides based on strategy (mirroring _quantize logic)
    if args is not None and args.strategy == QuantizationStrategy.BLOCK:
        input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x_strides
        (
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
        ) = out_strides
    elif args is not None and args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        input_stride_0 = 0
        input_stride_1, input_stride_2, input_stride_3 = x_strides
        output_stride_0 = 0
        output_stride_1, output_stride_2, output_stride_3 = out_strides
    elif args is not None and args.strategy in (
        QuantizationStrategy.TENSOR,
        QuantizationStrategy.CHANNEL,
    ):
        input_stride_0 = 0
        input_stride_1, input_stride_3 = x_strides
        input_stride_2 = 0
        output_stride_0 = 0
        output_stride_1, output_stride_3 = out_strides
        output_stride_2 = 0
    else:
        # Fallback: infer from tensor shape
        if x_q.ndim == 4:
            input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x_strides
            (
                output_stride_0,
                output_stride_1,
                output_stride_2,
                output_stride_3,
            ) = out_strides
        elif x_q.ndim == 3:
            input_stride_0 = 0
            input_stride_1, input_stride_2, input_stride_3 = x_strides
            output_stride_0 = 0
            output_stride_1, output_stride_2, output_stride_3 = out_strides
        else:
            input_stride_0 = 0
            input_stride_1, input_stride_3 = x_strides
            input_stride_2 = 0
            output_stride_0 = 0
            output_stride_1, output_stride_3 = out_strides
            output_stride_2 = 0

    with torch.get_device_module().device(x_q.device):
        _dequantize_kernel[grid](
            dequant_value,
            x_q,
            scale,
            zero_point if zero_point is not None else x_q,  # dummy pointer
            global_scale if global_scale is not None else x_q,  # dummy pointer
            input_stride_0,
            input_stride_1,
            input_stride_2,
            input_stride_3,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
            dim_0,
            dim_1,
            dim_2,
            dim_3,
            group_size,
            num_scale_cols,
            BLOCK_SIZE_R=block_size_r,
            BLOCK_SIZE_C=block_size_c,
            has_zero_point=zero_point is not None,
            has_global_scale=global_scale is not None,
        )

    dequant_value = dequant_value.reshape(original_shape)

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value


@torch.no_grad()
@ImplBackend.entrypoint("_dequantize")
def _dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
    args: QuantizationArgs | None = None,
) -> torch.Tensor:
    """
    Dequantize a tensor: output = (x_q - zero_point) * scale

    This is the PyTorch fallback implementation.
    Triton implementation is registered via ImplBackend.
    """
    if global_scale is not None:
        scale = scale / global_scale

    dequant_value = x_q.to(scale.dtype)

    # Ensure scale broadcasts correctly to x_q shape
    while scale.ndim < dequant_value.ndim:
        scale = scale.unsqueeze(-1)
    if zero_point is not None:
        while zero_point.ndim < dequant_value.ndim:
            zero_point = zero_point.unsqueeze(-1)

    if zero_point is not None:
        dequant_value = dequant_value - zero_point.to(scale.dtype)

    dequant_value = dequant_value * scale

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value
