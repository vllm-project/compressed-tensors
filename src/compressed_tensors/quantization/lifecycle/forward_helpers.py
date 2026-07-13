# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil

import torch
from compressed_tensors.quantization.quant_args import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationType,
    round_to_quantized_type_args,
)
from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant


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
    g_idx: torch.Tensor | None,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Group/tensor-group quantization: handle activation ordering, reshape
    into groups, quantize, restore."""
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

    # support column-order (default) quantization as well as other orderings
    # such as activation ordering. Below checks if g_idx has been initialized
    is_column_order = g_idx is None or g_idx.device.type == "meta" or -1 in g_idx
    if not is_column_order:
        perm = torch.argsort(g_idx)
        x = x.index_select(-1, perm)

    # reshape last dim into (num_groups, group_size)
    reshaped_dims = (ceil(x.shape[-1] / group_size), group_size)
    x = x.unflatten(-1, reshaped_dims)

    # print(f"x.shape: {x.shape}")
    # print(f"scale.shape: {scale.shape}")
    # print(f"zero_point.shape: {zero_point.shape if zero_point is not None else None}")
    # print(f"q_min.shape: {q_min.shape}")
    # print(f"q_max.shape: {q_max.shape}")
    # print(f"global_scale.shape: {global_scale.shape if global_scale is not None else None}")
    # print(f"x strides: {x.stride()}")
    # print(f"scale strides: {scale.stride()}")
    # print(f"zero_point strides: {zero_point.stride() if zero_point is not None else None}")

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

    if not is_column_order:
        inv_perm = torch.argsort(perm)
        output = output.index_select(-1, inv_perm)

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
    """
    # compute effective scale once
    if global_scale is not None:
        scale = scale / global_scale

    scaled = x / scale

    if zero_point is not None:
        scaled += zero_point.to(x.dtype)

    # clamp and round (stays in float — no int8/fp8 intermediate)
    quantized = round_to_quantized_type_args(
        tensor=scaled, args=args, min=q_min, max=q_max
    )

    # dequantize: subtract zero_point and multiply by scale
    # cast to scale.dtype to match _dequantize behavior
    dequant = quantized.to(scale.dtype)
    if zero_point is not None:
        dequant = dequant - zero_point.to(scale.dtype)

    return dequant * scale


import triton
import triton.language as tl

# Quantization type constants for Triton kernel
QUANT_TYPE_INT = tl.constexpr(0)
QUANT_TYPE_FLOAT = tl.constexpr(1)


@triton.jit
def _round_to_fp4(x):
    """
    Round float values to the nearest E2M1 representable value.
    FP4 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and their negatives)
    
    Matches the thresholds in the Python ``cast_to_fp4`` exactly.
    Based on vllm's nvfp4_emulation_utils.py implementation.
    """
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.abs(x)
    
    # Map to FP4 representable values based on thresholds
    # Start with default 0.0, then overwrite from highest to lowest threshold
    result = tl.where(abs_x > 5.0, 6.0, 0.0)
    result = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 4.0, result)
    result = tl.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, result)
    result = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, result)
    result = tl.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, result)
    result = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 1.0, result)
    result = tl.where((abs_x > 0.25) & (abs_x < 0.75), 0.5, result)
    
    return result * sign


@triton.jit
def _quantize_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    q_min_ptr: tl.tensor,
    q_max_ptr: tl.tensor,
    global_scale_ptr: tl.tensor,
    num_rows,
    num_cols,
    group_size,
    # quant_type: QUANT_TYPE_INT or QUANT_TYPE_FLOAT
    quant_type: tl.constexpr,
    # num_bits: for FLOAT quantization (4 or 8)
    num_bits: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Set up the pids.
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    offsets_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offsets_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offsets = num_cols * offsets_r[:, None] + offsets_c[None, :]

    masks_r = offsets_r < num_rows
    masks_c = offsets_c < num_cols
    masks = masks_r[:, None] & masks_c[None, :]

    scale_offsets_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    scale_offsets_c = (pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C )) // group_size
    scale_offsets = (num_cols // group_size) * scale_offsets_r[:, None] + scale_offsets_c[None, :]
    scale_masks_r = scale_offsets_r < num_rows
    scale_masks_c = scale_offsets_c < num_cols // group_size
    scale_masks = scale_masks_r[:, None] & scale_masks_c[None, :]

    result_offsets_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    result_offsets_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    result_offsets = (
        num_cols * result_offsets_r[:, None] + result_offsets_c[None, :]
    )

    result_masks_r = result_offsets_r < num_rows
    result_masks_c = result_offsets_c < num_cols
    result_masks = result_masks_r[:, None] & result_masks_c[None, :]

    input = tl.load(input_ptr + offsets, masks, 0.0)
    scale = tl.load(scale_ptr + scale_offsets, scale_masks, 0.0)

    if global_scale_ptr is not None:
        global_scale = tl.load(global_scale_ptr)
        scale = scale / global_scale.to(scale.dtype)

    output = input / scale

    if zero_point_ptr is not None:
        zero_point = tl.load(zero_point_ptr + scale_offsets, scale_masks, 0.0)
        output += zero_point
    
    # clamp and round (equivalent to round_to_quantized_type_args)
    q_min = tl.load(q_min_ptr)
    q_max = tl.load(q_max_ptr)
    
    if quant_type == QUANT_TYPE_INT:
        # INT quantization: round half away from zero, then clamp
        # Based on vllm's triton_reshape_and_cache_flash.py approach
        output = tl.where(output >= 0, output + 0.5, output - 0.5)
        output = tl.clamp(output, q_min, q_max)
    elif quant_type == QUANT_TYPE_FLOAT:
        # FLOAT quantization: clamp first, then round to representable values
        output = tl.clamp(output, q_min, q_max)
        if num_bits == 4:
            # FP4 E2M1 rounding (based on vllm's nvfp4_emulation_utils.py)
            output = _round_to_fp4(output)
        # Note: FP8 would require hardware support or casting, not implemented here
        
    tl.store(output_ptr + result_offsets, output, result_masks)


@torch.no_grad()
def _quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:

    # Triton only works with CUDA tensors
    do_triton: bool = x.is_cuda

    if not do_triton:
        # if a global scale is optionally provided, use it
        # to further scale the local `scale` parameter
        scale_ground = scale.clone()
        if global_scale is not None:
            scale_ground = scale / global_scale

        scaled = x / scale_ground
        if zero_point is not None:
            scaled += zero_point.to(x.dtype)
        quantized_ground = round_to_quantized_type_args(
            tensor=scaled, args=args, min=q_min, max=q_max
        )
        # quantized_ground = scaled
        if dtype is not None:
            quantized_ground = quantized_ground.to(dtype)
        return quantized_ground

    original_shape = x.shape
    
    # Handle both 2D and 3D tensors:
    # - 3D [rows, groups, group_size]: group_size = shape[2]
    # - 2D [rows, cols]: treat as single group, group_size = cols
    if x.ndim == 3:
        group_size = x.shape[2]
        x = x.reshape(x.shape[0], -1)
        scale = scale.reshape(scale.shape[0], -1)
        if zero_point is not None:
            zero_point = zero_point.reshape(zero_point.shape[0], -1)
    elif x.ndim == 2:
        group_size = x.shape[1]  # Entire row is one "group"
        num_rows = x.shape[0]
        # Kernel expects scale shape [num_rows, num_groups] where num_groups = 1 for 2D
        # Expand scale to [num_rows, 1] so kernel can index by row
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
        raise ValueError(f"Expected 2D or 3D tensor, got {x.ndim}D")

    block_size_r: int = 32
    block_size_c: int = 32
    num_rows = x.shape[0]
    num_cols = x.shape[1]
    grid = lambda META: (
        triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
        triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
    )
    quantized_value = torch.empty_like(x)

    # Determine quantization type
    quant_type = QUANT_TYPE_INT if args.type == QuantizationType.INT else QUANT_TYPE_FLOAT
    num_bits = args.num_bits
    
    _quantize_kernel[grid](quantized_value,
                           x,
                           scale,
                           zero_point,
                           q_min,
                           q_max,
                           global_scale,
                           num_rows,
                           num_cols,
                           group_size,
                           quant_type=quant_type,
                           num_bits=num_bits,
                           BLOCK_SIZE_R=block_size_r,
                           BLOCK_SIZE_C=block_size_c)

    quantized_value = quantized_value.reshape(original_shape)

    # Rounding is done inside _quantize_kernel for INT and FP4 types.
    # For FP8, apply rounding via dtype cast (not supported in Triton kernel).
    if args.type == QuantizationType.FLOAT and args.num_bits == 8:
        original_dtype = quantized_value.dtype
        quantized_value = quantized_value.to(FP8_E4M3_DATA.dtype).to(original_dtype)

    if dtype is not None:
        quantized_value = quantized_value.to(dtype)

    return quantized_value


@torch.no_grad()
def _dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    # if a global scale is optionally provided, use it
    # to further scale the local `scale` parameter
    if global_scale is not None:
        scale = scale / global_scale

    dequant_value = x_q.to(scale.dtype)

    if zero_point is not None:
        dequant_value = dequant_value - zero_point.to(scale.dtype)

    dequant_value = dequant_value * scale

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value
