# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil

import torch


try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    _triton_available = False

from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    round_to_quantized_type_args,
)
from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant
from compressed_tensors.quantization.utils.fp4_utils import _round_to_fp4


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
        # Determine if Triton should be used
        is_gpu = x.is_cuda or x.is_xpu
        is_fp8 = _needs_fp8(x, scale, zero_point, global_scale, args=args)
        fp8_hw_ok = _is_fp8_supported(x.device) if is_fp8 else True
        do_triton: bool = is_gpu and (not is_fp8 or fp8_hw_ok)
        # do_triton = False

        # Adapt scale/zp for Triton if needed
        if do_triton and args.strategy in (
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
        ):
            num_rows = x.shape[0]
            scale, zero_point = adapt_scale_and_zp_for_triton(
                scale, zero_point, num_rows
            )

        # print("do quantize with:")
        # print("   x.shape:", x.shape, "x.stride:", x.stride())
        # print("   scale.shape:", scale.shape, "scale.stride:", scale.stride())
        # print("   zero_point.shape:", zero_point.shape if zero_point is not None else None, "zero_point.stride:", zero_point.stride() if zero_point is not None else None)
        # print("   q_min.shape:", q_min.shape, "q_min.stride:", q_min.stride())
        # print("   q_max.shape:", q_max.shape, "q_max.stride:", q_max.stride())
        # print("   args:", args)
        # print("   dtype:", dtype if dtype is not None else None)
        # print("   global_scale.shape:", global_scale.shape if global_scale is not None else None, "global_scale.stride:", global_scale.stride() if global_scale is not None else None)
        # print("   do_triton:", do_triton)

        # x = x.contiguous()
        # scale = scale.contiguous()
        # zero_point = zero_point.contiguous() if zero_point is not None else None

        return _quantize(
            x=x,
            scale=scale,
            zero_point=zero_point,
            q_min=q_min,
            q_max=q_max,
            args=args,
            dtype=dtype,
            global_scale=global_scale,
            do_triton=do_triton,
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


# Quantization type constants for Triton kernel
QUANT_TYPE_INT = tl.constexpr(0)
QUANT_TYPE_FLOAT = tl.constexpr(1)

if _triton_available:

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
        quant_type: tl.constexpr,  # QUANT_TYPE_INT or QUANT_TYPE_FLOAT
        num_bits: tl.constexpr,  # 4 or 8
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
        scale_offsets_c = (
            pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
        ) // group_size
        scale_offsets = (num_cols // group_size) * scale_offsets_r[
            :, None
        ] + scale_offsets_c[None, :]
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
            output = tl.clamp(output, q_min, q_max)
            output = tl.extra.cuda.libdevice.rint(output)
        elif quant_type == QUANT_TYPE_FLOAT:
            output = tl.clamp(output, q_min, q_max)
            if num_bits == 4:
                # Convert to bfloat16 for FP4 rounding (matches CPU path)
                orig_dtype = output.dtype
                output = _round_to_fp4(output.to(tl.bfloat16)).to(orig_dtype)
            elif num_bits == 8:
                output = output.to(tl.float8e4nv).to(output.dtype)

        tl.store(output_ptr + result_offsets, output, result_masks)

    @triton.jit
    def _quantize_kernel_strided(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        scale_ptr: tl.tensor,
        zero_point_ptr: tl.tensor,
        q_min_ptr: tl.tensor,
        q_max_ptr: tl.tensor,
        global_scale_ptr: tl.tensor,
        # Input tensor strides (4D, pad with 0 for fewer dims)
        input_stride_0,
        input_stride_1,
        input_stride_2,
        input_stride_3,
        # Scale tensor strides for each index dimension
        # scale_offset = idx_0 * scale_stride_idx0 + idx_1 * scale_stride_idx1 + idx_2 * scale_stride_idx2
        # Set unused strides to 0
        scale_stride_idx0,
        scale_stride_idx1,
        scale_stride_idx2,
        # Output tensor strides (4D, pad with 0 for fewer dims)
        output_stride_0,
        output_stride_1,
        output_stride_2,
        output_stride_3,
        # Tensor dimensions (4D shape, pad with 1 for fewer dims)
        # dims 0,1 make up "rows", dims 2,3 make up "cols"
        dim_0,
        dim_1,
        dim_2,
        dim_3,
        # Scale dimensions for masking (corresponding to which idx dims are used)
        scale_dim_0,
        scale_dim_1,
        scale_dim_2,
        quant_type: tl.constexpr,  # QUANT_TYPE_INT or QUANT_TYPE_FLOAT
        num_bits: tl.constexpr,  # 4 or 8
        BLOCK_SIZE_R: tl.constexpr,
        BLOCK_SIZE_C: tl.constexpr,
    ):
        """General quantize kernel for non-contiguous tensors using explicit strides.

        Handles tensors up to 4D by treating them as a 2D view:
        - row indices span dim_0 * dim_1
        - col indices span dim_2 * dim_3

        Scale indexing is flexible via scale_stride_idx{0,1,2} parameters:
        - BLOCK [n_rb, n_cb, bh, bw]: scale indexed by (idx_0, idx_1)
        - GROUP [1, batch, num_groups, gs]: scale indexed by (idx_1, idx_2)
        - TENSOR [1, rows, 1, cols]: scale indexed by (idx_1)
        """
        pid_r = tl.program_id(axis=0)
        pid_c = tl.program_id(axis=1)

        # Tile indices in the flattened 2D view
        tile_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
        tile_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

        # Convert flattened row index to (idx_0, idx_1)
        idx_0 = tile_r // dim_1
        idx_1 = tile_r % dim_1

        # Convert flattened col index to (idx_2, idx_3)
        idx_2 = tile_c // dim_3
        idx_3 = tile_c % dim_3

        # Compute 4D offsets using actual strides
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

        # Scale indexed by combination of idx_0, idx_1, idx_2 (configurable via strides)
        scale_offsets = (
            idx_0[:, None] * scale_stride_idx0
            + idx_1[:, None] * scale_stride_idx1
            + idx_2[None, :] * scale_stride_idx2
        )

        # Masks for valid indices
        masks_0 = idx_0 < dim_0
        masks_1 = idx_1 < dim_1
        masks_2 = idx_2 < dim_2
        masks_3 = idx_3 < dim_3
        masks = masks_0[:, None] & masks_1[:, None] & masks_2[None, :] & masks_3[None, :]

        # Scale masks for whichever dimensions are used
        scale_masks_0 = idx_0 < scale_dim_0
        scale_masks_1 = idx_1 < scale_dim_1
        scale_masks_2 = idx_2 < scale_dim_2
        scale_masks = scale_masks_0[:, None] & scale_masks_1[:, None] & scale_masks_2[None, :]

        # Load input and scale
        input = tl.load(input_ptr + input_offsets, masks, 0.0)
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
            output = tl.clamp(output, q_min, q_max)
            output = tl.extra.cuda.libdevice.rint(output)
        elif quant_type == QUANT_TYPE_FLOAT:
            output = tl.clamp(output, q_min, q_max)
            if num_bits == 4:
                orig_dtype = output.dtype
                output = _round_to_fp4(output.to(tl.bfloat16)).to(orig_dtype)
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
        # Intel XPU: conservatively disable FP8 in Triton
        return False
    return False


def adapt_scale_and_zp_for_triton(
    scale: torch.Tensor, zero_point: torch.Tensor | None, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Adapt scale and zero point for Triton kernel.
    This is required when we use group strategies, so that Triton
    can read the correct scale and zero point for each group.

    Note: We keep scale/zp contiguous because:
    1. They are small tensors (one value per row/group), so contiguous() is cheap
    2. The strided kernel focuses on handling large non-contiguous input tensors
    """
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
    return scale, zero_point


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
    do_triton: bool = False,
) -> torch.Tensor:

    if not _triton_available or not do_triton:
        # if a global scale is optionally provided, use it
        # to further scale the local `scale` parameter
        if global_scale is not None:
            scale /= global_scale

        scaled = x / scale
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

    # Determine quantization type
    quant_type = (
        QUANT_TYPE_INT if args.type == QuantizationType.INT else QUANT_TYPE_FLOAT
    )
    num_bits = args.num_bits

    # Check if we need the strided kernel for non-contiguous input tensors
    use_strided_kernel = not x.is_contiguous()

    if args.strategy == QuantizationStrategy.BLOCK:
        # Block quantization - 4D input with potentially non-contiguous strides
        n_rb, n_cb, bh, bw = x.shape
        num_rows = n_rb * n_cb
        num_cols = bh * bw

        block_size_r: int = 32
        block_size_c: int = 32

        def grid(META):
            return (
                triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
                triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
            )

        quantized_value = torch.empty_like(x)

        # Get actual strides for non-contiguous tensor support
        input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x.stride()
        output_stride_0, output_stride_1, output_stride_2, output_stride_3 = (
            quantized_value.stride()
        )

        # BLOCK: scale indexed by (idx_0, idx_1) = (n_rb, n_cb)
        # scale shape: [n_rb, n_cb, 1, 1]
        scale_stride_idx0 = scale.stride()[0]
        scale_stride_idx1 = scale.stride()[1]
        scale_stride_idx2 = 0  # idx_2 not used for scale

        _quantize_kernel_strided[grid](
            quantized_value,
            x,
            scale,
            zero_point,
            q_min,
            q_max,
            global_scale,
            input_stride_0,
            input_stride_1,
            input_stride_2,
            input_stride_3,
            scale_stride_idx0,
            scale_stride_idx1,
            scale_stride_idx2,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
            n_rb,
            n_cb,
            bh,
            bw,
            scale.shape[0],  # scale_dim_0 for idx_0
            scale.shape[1],  # scale_dim_1 for idx_1
            bh,  # scale_dim_2 for idx_2 (always valid since not used)
            quant_type=quant_type,
            num_bits=num_bits,
            BLOCK_SIZE_R=block_size_r,
            BLOCK_SIZE_C=block_size_c,
        )
    elif args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        # Group quantization - 3D input (batch, num_groups, group_size)
        group_size = x.shape[2]
        num_rows = x.shape[0]
        num_cols = x.shape[1] * x.shape[2]

        block_size_r: int = 32
        block_size_c: int = 32

        def grid(META):
            return (
                triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
                triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
            )

        quantized_value = torch.empty_like(x)

        if use_strided_kernel:
            # Use strided kernel for non-contiguous tensors
            # 3D [batch, num_groups, group_size] -> 4D [1, batch, num_groups, group_size]
            input_stride_0, input_stride_1, input_stride_2 = x.stride()
            output_stride_0, output_stride_1, output_stride_2 = quantized_value.stride()

            # GROUP: scale indexed by (idx_1, idx_2) = (batch, num_groups)
            # scale shape: [batch, num_groups, 1]
            scale_stride_idx0 = 0  # idx_0 not used (always 0)
            scale_stride_idx1 = scale.stride()[0]  # batch dimension
            scale_stride_idx2 = scale.stride()[1]  # num_groups dimension

            _quantize_kernel_strided[grid](
                quantized_value,
                x,
                scale,
                zero_point,
                q_min,
                q_max,
                global_scale,
                0,  # input_stride_0 (dim 0 is padded with 1)
                input_stride_0,
                input_stride_1,
                input_stride_2,
                scale_stride_idx0,
                scale_stride_idx1,
                scale_stride_idx2,
                0,  # output_stride_0 (dim 0 is padded with 1)
                output_stride_0,
                output_stride_1,
                output_stride_2,
                1,  # dim_0 (padded)
                x.shape[0],  # dim_1 = batch
                x.shape[1],  # dim_2 = num_groups
                x.shape[2],  # dim_3 = group_size
                1,  # scale_dim_0 for idx_0 (always valid since padded)
                scale.shape[0],  # scale_dim_1 for idx_1
                scale.shape[1],  # scale_dim_2 for idx_2
                quant_type=quant_type,
                num_bits=num_bits,
                BLOCK_SIZE_R=block_size_r,
                BLOCK_SIZE_C=block_size_c,
            )
        else:
            _quantize_kernel[grid](
                quantized_value,
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
                BLOCK_SIZE_C=block_size_c,
            )

        quantized_value = quantized_value.reshape(original_shape)
    elif args.strategy in (QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL):
        # Tensor/Channel quantization - 2D input
        group_size = x.shape[1]
        num_rows = x.shape[0]
        num_cols = x.shape[1]

        block_size_r: int = 32
        block_size_c: int = 32

        def grid(META):
            return (
                triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
                triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
            )

        quantized_value = torch.empty_like(x)

        if use_strided_kernel:
            # Use strided kernel for non-contiguous tensors
            # 2D [rows, cols] -> 4D [1, rows, 1, cols]
            input_stride_0, input_stride_1 = x.stride()
            output_stride_0, output_stride_1 = quantized_value.stride()

            # TENSOR: scale indexed by idx_1 only (rows)
            # scale shape: [rows, 1] (after adapt_scale_and_zp_for_triton)
            scale_stride_idx0 = 0  # idx_0 not used (always 0)
            scale_stride_idx1 = scale.stride()[0]  # row dimension
            scale_stride_idx2 = 0  # idx_2 not used (always 0)

            _quantize_kernel_strided[grid](
                quantized_value,
                x,
                scale,
                zero_point,
                q_min,
                q_max,
                global_scale,
                0,  # input_stride_0 (dim 0 is padded with 1)
                input_stride_0,
                0,  # input_stride_2 (dim 2 is padded with 1)
                input_stride_1,
                scale_stride_idx0,
                scale_stride_idx1,
                scale_stride_idx2,
                0,  # output_stride_0 (dim 0 is padded with 1)
                output_stride_0,
                0,  # output_stride_2 (dim 2 is padded with 1)
                output_stride_1,
                1,  # dim_0 (padded)
                x.shape[0],  # dim_1 = rows
                1,  # dim_2 (padded)
                x.shape[1],  # dim_3 = cols
                1,  # scale_dim_0 for idx_0 (always valid since padded)
                scale.shape[0],  # scale_dim_1 for idx_1
                1,  # scale_dim_2 for idx_2 (always valid since padded)
                quant_type=quant_type,
                num_bits=num_bits,
                BLOCK_SIZE_R=block_size_r,
                BLOCK_SIZE_C=block_size_c,
            )
        else:
            _quantize_kernel[grid](
                quantized_value,
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
                BLOCK_SIZE_C=block_size_c,
            )

        quantized_value = quantized_value.reshape(original_shape)
    else:
        raise ValueError(f"Unsupported quantization strategy: {args.strategy}")

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
