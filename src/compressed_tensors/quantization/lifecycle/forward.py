# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import wraps

import torch
from compressed_tensors.quantization.lifecycle.forward_helpers import (
    _apply_quantize_op,
    _process_block,
    _process_group,
)
from compressed_tensors.quantization.quant_args import (
    DynamicType,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import (
    calculate_range,
    compute_dynamic_scales_and_zp,
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

    if args.strategy == QuantizationStrategy.BLOCK:
        return _process_block(
            x,
            scale,
            zero_point,
            args,
            q_min,
            q_max,
            dtype,
            do_quantize,
            do_dequantize,
            global_scale,
        )
    elif args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        return _process_group(
            x,
            scale,
            zero_point,
            args,
            q_min,
            q_max,
            dtype,
            do_quantize,
            do_dequantize,
            g_idx,
            global_scale,
        )
    else:
        # covers tensor, channel, token, and attn_head strategies
        return _apply_quantize_op(
            x,
            scale,
            zero_point,
            q_min,
            q_max,
            args,
            dtype,
            do_quantize,
            do_dequantize,
            global_scale,
        )


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


