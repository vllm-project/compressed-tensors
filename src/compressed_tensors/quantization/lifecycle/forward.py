# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import wraps
from math import ceil

import torch
from compressed_tensors.quantization.quant_args import (
    DynamicType,
    QuantizationArgs,
    QuantizationStrategy,
    round_to_quantized_type_args,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import (
    calculate_range,
    compute_dynamic_scales_and_zp,
    maybe_pad_tensor_for_block_quant,
)
from compressed_tensors.utils import patch_attr
from torch.nn import Module


__all__ = [
    "quantize",
    "dequantize",
    "fake_quantize",
    "set_forward_quantized",
    "forward_quantize",
]


@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    dtype: torch.dtype | None = None,
    g_idx: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Quantize the input tensor x using the QuantizationStrategy specified in args.
    Quantization can be done per tensor, channel, token or group. For group
    quantization, the group_size must be divisible by the column size. The input scale
    and zero_points are reshaped to support vectorization (Assumes 1 is the
    channel dimension)

    :param x: Input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args dictating how to quantize x
    :param dtype: optional dtype to cast the quantized output to
    :param g_idx: optional mapping from column index to group index
    :param global_scale: optional constant to scale the quantization scale during QDQ
    :return: fake quantized tensor
    """

    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        dtype=dtype,
        do_quantize=True,
        do_dequantize=False,
        g_idx=g_idx,
        global_scale=global_scale,
    )


@torch.no_grad()
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    args: QuantizationArgs | None = None,
    dtype: torch.dtype | None = None,
    g_idx: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Dequantize a quantized input tensor x_q based on the strategy specified in args. If
    args is not provided, the strategy will be inferred.

    :param x: quantized input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args used to quantize x_q
    :param dtype: optional dtype to cast the dequantized output to
    :param g_idx: optional mapping from column index to group index
    :param global_scale: optional constant to scale the quantization scale during QDQ
    :return: dequantized float tensor
    """
    if args is None:
        if scale.ndim == 0 or scale.ndim == 1:
            args = QuantizationArgs(strategy=QuantizationStrategy.TENSOR)
        elif scale.ndim == 2:
            if scale.shape[1] == 1:
                args = QuantizationArgs(strategy=QuantizationStrategy.CHANNEL)
            # Scale height matches input or is 1 -> group quantization across columns
            #
            # Example 1: scale.shape[0] == 1
            # x_q: (4, 8), scale: (1, 4) -> 2 columns per group
            #
            # Example 2: scale.shape[0] == x_q.shape[0]
            # x_q: (4, 8), scale: (4, 4) -> 2 elements per group (per row)
            elif (scale.shape[0] == 1) or (scale.shape[0] == x_q.shape[0]):
                group_size = int(x_q.shape[1] / scale.shape[1])
                args = QuantizationArgs(
                    strategy=QuantizationStrategy.GROUP, group_size=group_size
                )
            else:
                rows, cols = x_q.shape[-2], x_q.shape[-1]
                block_height = rows // scale.shape[0]  # Rows per block
                block_width = cols // scale.shape[1]  # Columns per block

                args = QuantizationArgs(
                    strategy=QuantizationStrategy.BLOCK,
                    block_structure=[block_height, block_width],
                )
        else:
            raise ValueError(
                f"Could not infer a quantization strategy from scale with {scale.ndim} "
                "dimmensions. Expected 0 or 2 dimmensions."
            )

    if dtype is None:
        dtype = scale.dtype

    return _process_quantization(
        x=x_q,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=False,
        do_dequantize=True,
        dtype=dtype,
        g_idx=g_idx,
        global_scale=global_scale,
    )


@torch.no_grad()
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    g_idx: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fake quantize the input tensor x by quantizing then dequantizing with
    the QuantizationStrategy specified in args. Quantization can be done per tensor,
    channel, token or group. For group quantization, the group_size must be divisible
    by the column size. The input scale  and zero_points are reshaped to support
    vectorization (Assumes 1 is the channel dimension)

    :param x: Input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args dictating how to quantize x
    :param g_idx: optional mapping from column index to group index
    :param global_scale: optional constant to scale the quantization scale during QDQ
    :return: fake quantized tensor
    """
    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=True,
        do_dequantize=True,
        g_idx=g_idx,
        global_scale=global_scale,
    )


@torch.no_grad()
def _process_quantization(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    g_idx: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    do_quantize: bool = True,
    do_dequantize: bool = True,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    q_min, q_max = calculate_range(args, x.device)
    group_size = args.group_size

    # blockwise FP8: quantize per 2D block, supports block_structure for static block
    # quantization
    if args.strategy == QuantizationStrategy.BLOCK:
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

        # expand scale/zero_point for blocks
        sb = scale.unsqueeze(-1).unsqueeze(-1)
        zb = zero_point.unsqueeze(-1).unsqueeze(-1) if zero_point is not None else None
        if do_quantize and do_dequantize:
            x_blocks = _quantize_dequantize(
                x=x_blocks,
                scale=sb,
                zero_point=zb,
                q_min=q_min,
                q_max=q_max,
                args=args,
                global_scale=global_scale,
            )
        elif do_quantize:
            x_blocks = _quantize(
                x=x_blocks,
                scale=sb,
                zero_point=zb,
                q_min=q_min,
                q_max=q_max,
                args=args,
                dtype=dtype,
                global_scale=global_scale,
            )
        elif do_dequantize:
            x_blocks = _dequantize(
                x_q=x_blocks,
                scale=sb,
                zero_point=zb,
                global_scale=global_scale,
            )
        # restore padded shape
        output = x_blocks.transpose(1, 2).reshape(padded_shape)
        # truncate to original dimensions if padding was applied
        if original_shape != padded_shape:
            output = output[tuple([slice(v) for v in original_shape])]
    elif args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        output_dtype = dtype if dtype is not None else x.dtype
        output = torch.zeros_like(x).to(output_dtype)
        columns = output.shape[-1]

        # TODO: make validation step for inputs

        while scale.ndim < 2:
            # pad scale and zero point dims for slicing
            scale = scale.unsqueeze(1)
            zero_point = zero_point.unsqueeze(1) if zero_point is not None else None

        if columns >= group_size:
            if columns % group_size != 0:
                raise ValueError(
                    "tensor column shape must be divisble "
                    f"by the given group_size {group_size} but got {columns}"
                )

        # support column-order (default) quantization as well as other orderings
        # such as activation ordering. Below checks if g_idx has been initialized
        is_column_order = g_idx is None or g_idx.device.type == "meta" or -1 in g_idx
        if is_column_order:
            num_groups = int(ceil(columns / group_size))
            group_sizes = torch.full((num_groups,), group_size, dtype=torch.int)

        else:
            group_indices, group_sizes = torch.unique(g_idx, return_counts=True)
            group_sizes = group_sizes[torch.argsort(group_indices)]

            perm = torch.argsort(g_idx)
            x = x.index_select(-1, perm)

        # Maintain all dimensions except the last dim, which is divided by group_size
        reshaped_dims = (
            ceil(x.shape[-1] / group_size),
            group_size,
        )
        x = x.unflatten(-1, reshaped_dims)

        if do_quantize and do_dequantize:
            output = _quantize_dequantize(
                x=x,
                scale=scale.unsqueeze(-1),
                zero_point=zero_point.unsqueeze(-1) if zero_point is not None else None,
                q_min=q_min,
                q_max=q_max,
                args=args,
                global_scale=global_scale,
            )
        elif do_quantize:
            output = _quantize(
                x=x,
                scale=scale.unsqueeze(-1),
                zero_point=zero_point.unsqueeze(-1) if zero_point is not None else None,
                dtype=dtype,
                global_scale=global_scale,
                q_min=q_min,
                q_max=q_max,
                args=args,
            )
        elif do_dequantize:
            output = _dequantize(
                x_q=x,
                scale=scale.unsqueeze(-1),
                zero_point=zero_point.unsqueeze(-1) if zero_point is not None else None,
                global_scale=global_scale,
            )

        output = output.flatten(start_dim=-2)
        output = output.to(output_dtype)

        if not is_column_order:
            inv_perm = torch.argsort(perm)
            output = output.index_select(-1, inv_perm)

    else:  # covers tensor, channel, token, and attn_head strategies
        if do_quantize and do_dequantize:
            output = _quantize_dequantize(
                x=x,
                scale=scale,
                zero_point=zero_point,
                q_min=q_min,
                q_max=q_max,
                args=args,
                global_scale=global_scale,
            )
        elif do_quantize:
            output = _quantize(
                x=x,
                scale=scale,
                zero_point=zero_point,
                q_min=q_min,
                q_max=q_max,
                args=args,
                dtype=dtype,
                global_scale=global_scale,
            )
        elif do_dequantize:
            output = _dequantize(
                x,
                scale=scale,
                zero_point=zero_point,
                global_scale=global_scale,
            )

    return output


def set_forward_quantized(module: torch.nn.Linear | torch.nn.Embedding):
    """
    Replace a linear or embedding module's forward function with one that performs
    on-the-fly QDQ. Note that weight quantiation will be skipped for compressed modules.

    All QDQ operations can be skipped by setting `module.quantization_enabled = False`

    :param module: linear or embedding module whose forward function will be replaced
    """

    @wraps(module.forward.__func__)
    def quantized_forward(
        self: torch.nn.Linear | torch.nn.Embedding, input: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantized forward pass of a linear or embedding module

        :param self: instance of linear or embedding module
        :param input: input activations to this module
        :return: linear or embedding output
        """
        scheme: QuantizationScheme | None = getattr(self, "quantization_scheme", None)
        status: QuantizationStatus | None = getattr(self, "quantization_status", None)
        enabled: bool = (
            getattr(self, "quantization_enabled", True)
            and scheme is not None
            and status is not None
        )
        weight = self.weight  # onload once
        weight_data = weight.data

        if enabled and scheme.input_activations:
            input = forward_quantize(self, input, "input", scheme.input_activations)

        if enabled and scheme.weights and status < QuantizationStatus.COMPRESSED:
            weight_data = forward_quantize(self, weight_data, "weight", scheme.weights)

        with patch_attr(weight, "data", weight_data):
            output = self.__class__.forward(self, input)

        if enabled and scheme.output_activations:
            output = forward_quantize(self, output, "output", scheme.output_activations)

        return output

    module.forward = quantized_forward.__get__(module)


def forward_quantize(
    module: Module, value: torch.Tensor, base_name: str, args: "QuantizationArgs"
) -> torch.Tensor:
    # in compressed mode, the weight is already compressed and quantized so we don't
    # need to run fake quantization
    # TODO: remove this line, as this is already guarded by `set_forward_quantized`
    if (
        module.quantization_status >= QuantizationStatus.COMPRESSED
        and base_name == "weight"
    ):
        return value

    if value.numel() == 0:
        # if the tensor is empty,
        # skip quantization
        return value

    g_idx = getattr(module, "weight_g_idx", None)
    global_scale = getattr(module, f"{base_name}_global_scale", None)

    if args.dynamic in (True, DynamicType.LOCAL):
        # dynamic quantization - determine the scale/zp on the fly
        scale, zero_point = compute_dynamic_scales_and_zp(
            value=value, args=args, module=module, global_scale=global_scale
        )
    else:
        # static quantization - get scale and zero point from layer
        scale = getattr(module, f"{base_name}_scale")
        zero_point = getattr(module, f"{base_name}_zero_point", None)

    return fake_quantize(
        x=value,
        scale=scale,
        zero_point=zero_point,
        args=args,
        g_idx=g_idx,
        global_scale=global_scale,
    )


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
    # if a global scale is optionally provided, use it
    # to further scale the local `scale` parameter
    if global_scale is not None:
        scale = scale / global_scale

    scaled = x / scale

    if zero_point is not None:
        scaled += zero_point.to(x.dtype)

    # clamp and round
    quantized_value = round_to_quantized_type_args(
        tensor=scaled, args=args, min=q_min, max=q_max
    )

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
