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
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton, triton_req


# Quantization type constants for Triton kernel
QUANT_TYPE_INT = tl.constexpr(0)
QUANT_TYPE_FLOAT = tl.constexpr(1)


@triton.jit
def _quantize_dequantize_scalar_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    q_min_ptr: tl.tensor,
    q_max_ptr: tl.tensor,
    n_elements,
    HAS_ZERO_POINT: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused quantize-dequantize kernel for per-tensor (scalar) scale.
    Performs: output = ((clamp(round(x / scale + zp)) - zp) * scale

    This is the fast path for TENSOR strategy quantization.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and scale (scale is scalar)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    q_min = tl.load(q_min_ptr)
    q_max = tl.load(q_max_ptr)

    # Quantize: x / scale + zero_point, then clamp and round
    scaled = x / scale

    if HAS_ZERO_POINT:
        zero_point = tl.load(zero_point_ptr)
        scaled = scaled + zero_point

    # Clamp and round based on quantization type
    if QUANT_TYPE == QUANT_TYPE_INT:
        quantized = tl.clamp(scaled, q_min, q_max)
        quantized = tl.extra.cuda.libdevice.rint(quantized)
    else:  # QUANT_TYPE_FLOAT
        quantized = tl.clamp(scaled, q_min, q_max)
        if NUM_BITS == 4:
            quantized = _round_to_fp4(quantized.to(tl.bfloat16))
        # FP8: no additional rounding needed after clamp

    # Dequantize: (quantized - zero_point) * scale
    # Note: for FP4, quantized is bfloat16 but scale is float32,
    # so the result will be promoted to float32
    if HAS_ZERO_POINT:
        dequantized = (quantized - zero_point) * scale
    else:
        dequantized = quantized * scale

    tl.store(output_ptr + offsets, dequantized, mask=mask)


@triton.jit
def _quantize_dequantize_grouped_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    global_scale_ptr: tl.tensor,
    q_min_ptr: tl.tensor,
    q_max_ptr: tl.tensor,
    num_rows,
    num_cols,
    group_size,
    QUANT_TYPE: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    has_zero_point: tl.constexpr,
    has_global_scale: tl.constexpr,
):
    """
    Fused quantize-dequantize kernel for per-group/channel scales.
    Performs: output = ((clamp(round(x / scale + zp)) - zp) * scale

    Handles per-group scale/zero_point with configurable group_size.
    """
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    offsets_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offsets_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offsets = num_cols * offsets_r[:, None] + offsets_c[None, :]

    masks_r = offsets_r < num_rows
    masks_c = offsets_c < num_cols
    masks = masks_r[:, None] & masks_c[None, :]

    # Scale indexing: maps input columns to scale columns via group_size
    scale_offsets_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    scale_offsets_c = (pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)) // group_size
    scale_offsets = (num_cols // group_size) * scale_offsets_r[
        :, None
    ] + scale_offsets_c[None, :]
    scale_masks_r = scale_offsets_r < num_rows
    scale_masks_c = scale_offsets_c < num_cols // group_size
    scale_masks = scale_masks_r[:, None] & scale_masks_c[None, :]

    # Load inputs
    x = tl.load(input_ptr + offsets, masks, 0.0)
    scale = tl.load(scale_ptr + scale_offsets, scale_masks, 0.0)
    q_min = tl.load(q_min_ptr)
    q_max = tl.load(q_max_ptr)

    # Apply global scale if present
    if has_global_scale:
        global_scale = tl.load(global_scale_ptr)
        scale = scale / global_scale.to(scale.dtype)

    # Quantize: x / scale + zero_point
    scaled = x / scale

    if has_zero_point:
        zero_point = tl.load(zero_point_ptr + scale_offsets, scale_masks, 0.0)
        scaled = scaled + zero_point

    # Clamp and round based on quantization type
    if QUANT_TYPE == QUANT_TYPE_INT:
        quantized = tl.clamp(scaled, q_min, q_max)
        quantized = tl.extra.cuda.libdevice.rint(quantized)
    else:  # QUANT_TYPE_FLOAT
        quantized = tl.clamp(scaled, q_min, q_max)
        if NUM_BITS == 4:
            quantized = _round_to_fp4(quantized.to(tl.bfloat16))

    # Dequantize: (quantized - zero_point) * scale
    # Note: for FP4, quantized is bfloat16 but scale is float32,
    # so the result will be promoted to float32
    if has_zero_point:
        output = (quantized - zero_point) * scale
    else:
        output = quantized * scale

    tl.store(output_ptr + offsets, output, masks)


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
        return _quantize_dequantize(
            x=x,
            scale=scale,
            zero_point=zero_point,
            q_min=q_min,
            q_max=q_max,
            args=args,
            global_scale=global_scale,
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


@torch.no_grad()
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
    Fused quantize-then-dequantize in a single pass, avoiding:
    - Double scale/global_scale division
    - Intermediate quantized dtype allocation

    Dispatches to Triton kernels on CUDA for better performance.
    """
    # Triton only works with CUDA and XPU tensors
    do_triton: bool = x.is_cuda or x.is_xpu

    # FP8 requires hardware dtype casting that Triton can't replicate,
    # so fall back to PyTorch ops for FP8 quantization
    is_fp8 = args.type == QuantizationType.FLOAT.value and args.num_bits == 8

    if not do_triton or is_fp8:
        # CPU fallback: use PyTorch ops
        effective_scale = scale
        if global_scale is not None:
            effective_scale = scale / global_scale

        scaled = x / effective_scale

        if zero_point is not None:
            scaled = scaled + zero_point.to(x.dtype)

        # clamp and round (stays in float — no int8/fp8 intermediate)
        quantized = round_to_quantized_type_args(
            tensor=scaled, args=args, min=q_min, max=q_max
        )

        # dequantize: subtract zero_point and multiply by scale
        dequant = quantized.to(effective_scale.dtype)
        if zero_point is not None:
            dequant = dequant - zero_point.to(effective_scale.dtype)

        return dequant * effective_scale

    # Check if we can use the fast scalar path (per-tensor scale)
    is_scalar_scale = scale.numel() == 1
    is_scalar_zp = zero_point is None or zero_point.numel() == 1

    if is_scalar_scale and is_scalar_zp and global_scale is None:
        return _quantize_dequantize_scalar(x, scale, zero_point, q_min, q_max, args)

    # Grouped path for per-group/channel scales
    return _quantize_dequantize_grouped(
        x, scale, zero_point, q_min, q_max, args, global_scale
    )


def _quantize_dequantize_scalar(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
) -> torch.Tensor:
    """Fast fused quantize-dequantize for per-tensor (scalar) scale."""
    original_shape = x.shape
    x_flat = x.flatten()

    n_elements = x_flat.numel()
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    output = torch.empty_like(x_flat)

    # Determine quantization type for kernel
    quant_type = (
        QUANT_TYPE_INT if args.type == QuantizationType.INT.value else QUANT_TYPE_FLOAT
    )
    num_bits = args.num_bits

    # Dummy pointer for zero_point if not provided
    zp_ptr = zero_point if zero_point is not None else scale

    with torch.get_device_module().device(x.device):
        _quantize_dequantize_scalar_kernel[grid](
            output,
            x_flat,
            scale,
            zp_ptr,
            q_min,
            q_max,
            n_elements,
            HAS_ZERO_POINT=zero_point is not None,
            QUANT_TYPE=quant_type,
            NUM_BITS=num_bits,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output.reshape(original_shape)


def _quantize_dequantize_grouped(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused quantize-dequantize for per-group/channel scales."""
    original_shape = x.shape

    # Convert to float for computation
    x = x.to(scale.dtype)

    # Handle different tensor dimensions (same logic as _dequantize_grouped)
    if x.ndim == 4:
        n_rb, n_cb, bh, bw = x.shape
        group_size = bh * bw
        x = x.reshape(n_rb * n_cb, bh * bw)
        scale = scale.reshape(n_rb * n_cb, 1)
        if zero_point is not None:
            zero_point = zero_point.reshape(n_rb * n_cb, 1)
    elif x.ndim == 3:
        group_size = x.shape[2]
        x = x.reshape(x.shape[0], -1)
        scale = scale.reshape(scale.shape[0], -1)
        if zero_point is not None:
            zero_point = zero_point.reshape(zero_point.shape[0], -1)
    elif x.ndim == 2:
        group_size = x.shape[1]
        num_rows = x.shape[0]
        if scale.ndim == 0:
            scale = scale.expand(num_rows, 1).contiguous()
        elif scale.ndim == 1:
            scale = scale.unsqueeze(1).expand(num_rows, 1).contiguous()
        elif scale.shape[0] == 1:
            scale = scale.expand(num_rows, -1).contiguous()
        if zero_point is not None:
            if zero_point.ndim == 0:
                zero_point = zero_point.expand(num_rows, 1).contiguous()
            elif zero_point.ndim == 1:
                zero_point = zero_point.unsqueeze(1).expand(num_rows, 1).contiguous()
            elif zero_point.shape[0] == 1:
                zero_point = zero_point.expand(num_rows, -1).contiguous()
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {x.ndim}D")

    block_size_r = 32
    block_size_c = 32
    num_rows = x.shape[0]
    num_cols = x.shape[1]

    def grid(META):
        return (
            triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
            triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
        )

    output = torch.empty_like(x)

    # Determine quantization type for kernel
    quant_type = (
        QUANT_TYPE_INT if args.type == QuantizationType.INT.value else QUANT_TYPE_FLOAT
    )
    num_bits = args.num_bits

    with torch.get_device_module().device(x.device):
        _quantize_dequantize_grouped_kernel[grid](
            output,
            x,
            scale,
            zero_point if zero_point is not None else x,  # dummy pointer
            global_scale if global_scale is not None else x,  # dummy pointer
            q_min,
            q_max,
            num_rows,
            num_cols,
            group_size,
            QUANT_TYPE=quant_type,
            NUM_BITS=num_bits,
            BLOCK_SIZE_R=block_size_r,
            BLOCK_SIZE_C=block_size_c,
            has_zero_point=zero_point is not None,
            has_global_scale=global_scale is not None,
        )

    return output.reshape(original_shape)


@triton.jit
def _dequantize_scalar_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    n_elements,
    HAS_ZERO_POINT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast Triton kernel for per-tensor dequantization with scalar scale.

    output = (x_q - zero_point) * scale

    Optimized for the common case where scale is a single scalar value.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x_q = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Load scalar scale (single value, broadcasted)
    scale = tl.load(scale_ptr)

    # Dequantize
    if HAS_ZERO_POINT:
        zero_point = tl.load(zero_point_ptr)
        output = (x_q - zero_point) * scale
    else:
        output = x_q * scale

    tl.store(output_ptr + offsets, output, mask=mask)


if HAS_TRITON:

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
        masks = (
            masks_0[:, None] & masks_1[:, None] & masks_2[None, :] & masks_3[None, :]
        )

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


@torch.no_grad()
def _dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
    args: QuantizationArgs | None = None,
) -> torch.Tensor:

    # Triton only works with CUDA and XPU tensors
    do_triton: bool = x_q.is_cuda or x_q.is_xpu

    if not do_triton or not HAS_TRITON:
        # CPU fallback
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

    # Check if we can use the fast scalar path (per-tensor scale)
    is_scalar_scale = scale.numel() == 1
    is_scalar_zp = zero_point is None or zero_point.numel() == 1

    if is_scalar_scale and is_scalar_zp and global_scale is None:
        # Fast path: use optimized scalar kernel
        return _dequantize_scalar(x_q, scale, zero_point, dtype)

    # Strided path: use group-aware kernel for per-group scales
    return _dequantize_grouped(x_q, scale, zero_point, dtype, global_scale, args)


def _dequantize_scalar(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fast dequantization for per-tensor (scalar) scale."""
    original_shape = x_q.shape
    x_q_float = x_q.to(scale.dtype).flatten()

    n_elements = x_q_float.numel()
    # Use large block size to minimize kernel launch overhead
    # For simple element-wise ops, fewer blocks = less overhead
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    dequant_value = torch.empty_like(x_q_float)

    # Dummy pointer for zero_point if not provided
    zp_ptr = zero_point if zero_point is not None else scale

    with torch.get_device_module().device(x_q.device):
        _dequantize_scalar_kernel[grid](
            dequant_value,
            x_q_float,
            scale,
            zp_ptr,
            n_elements,
            HAS_ZERO_POINT=zero_point is not None,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    dequant_value = dequant_value.reshape(original_shape)

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value


def _dequantize_grouped(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
    args: QuantizationArgs | None = None,
) -> torch.Tensor:
    """Dequantization for per-group scales using strided kernel."""
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
