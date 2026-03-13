# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
from compressed_tensors.quantization.quant_args import (
    BFLOAT16_DATA,
    FLOAT32_DATA,
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationType,
)


__all__ = [
    "maybe_convert_from_mx_exp",
    "generate_mx_scales",
    "round_to_power_2",
    "should_generate_mx_scales",
]

# Reference: https://github.com/vllm-project/vllm/blob/main/tests/quantization/reference_mxfp4.py # noqa: E501

# The exponent offset maps the group max into the quantized type's
# representable range.  It equals floor(log2(type_max)):
#   FP4 E2M1  max=6.0   -> floor(log2(6))   = 2
#   FP8 E4M3  max=448.0 -> floor(log2(448)) = 8
_MX_ELEM_OFFSET = {
    4: int(math.floor(math.log2(FP4_E2M1_DATA.max))),  # 2
    8: int(math.floor(math.log2(FP8_E4M3_DATA.max))),  # 8
}


def should_generate_mx_scales(args: QuantizationArgs):
    return (
        args.type == QuantizationType.FLOAT.value
        and args.group_size == 32
        and args.scale_dtype == torch.uint8
    )


def maybe_convert_from_mx_exp(
    args: QuantizationArgs, scale: torch.Tensor
) -> torch.Tensor:
    """
    Conditionally converts MX (MXFP4/MXFP8) scales from their E8M0 exponent
    format to float scales.

    If the quantization arguments indicate an MX format, the input `scale`
    is treated as E8M0 uint8 exponents and converted to float power-of-2
    scales. Otherwise, the input `scale` tensor is returned unchanged.

    :param args: quantization args to check for MX format
    :param scale: tensor of scale values (uint8 exponents for MX, or float)
    :return: float scale tensor, or original scale if not MX format
    """
    original_dtype = scale.dtype
    if should_generate_mx_scales(args):
        scale_exp = scale.to(torch.int32) - 127
        scale = 2.00 ** (scale_exp.to(torch.float))
        return scale.to(original_dtype)
    return scale


def round_to_power_2(x: torch.Tensor) -> torch.Tensor:
    """
    Round values to the closest power of 2.
    This is done by masking the values with BFLOAT16_SIGN_EXPONENT_MASK
    which essentially removes the mantissa and keeps the exponent.
    i.e the closest power of 2 for the input_value.

    E.g:
        0.0825 = 1.32 (mantissa) x 2**-4 (exponent)
        0.0825 ==> -4 (exponent) + 127 = 123 = 01111011 (8 bits for bfloat16)
        0.0825 ==> 0.32 (mantissa) = 0101001 (7 bits for bfloat16)
        0.0825 == 0b01111011_0101001 (bfloat16)
        0b01111011_0101001 & 111111111_0000000 == 0b01111011_0000000
        Keep the exponent + sign bit to give you the closest power of 2, 0.0625

    :param x: tensor to round to closest power of 2
    """
    scale_dtype = x.dtype
    if scale_dtype is torch.bfloat16:
        int_dtype = torch.uint16
        mantissa = BFLOAT16_DATA.mantissa
        exponent = BFLOAT16_DATA.exponent
    else:
        assert scale_dtype is torch.float32
        int_dtype = torch.uint32
        mantissa = FLOAT32_DATA.mantissa
        exponent = FLOAT32_DATA.exponent

    x = x.view(int_dtype).to(torch.int32)

    # Find closest power of 2
    VAL_TO_ADD = 1 << (mantissa - FP4_E2M1_DATA.mantissa - 1)
    # Add value to push the value to the next exponent
    SIGN_EXPONENT_MASK = ((1 << (exponent + 1)) - 1) << mantissa
    # mask to only keep exponent - we conservatively round down
    # to better represent smaller numbers / prevent overflow
    block_max_uint = torch.bitwise_and(x + VAL_TO_ADD, SIGN_EXPONENT_MASK)
    if scale_dtype is torch.bfloat16:
        return block_max_uint.to(int_dtype).view(scale_dtype)
    return block_max_uint.view(scale_dtype)


def generate_mx_scales(x: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """
    Generate MX scales (for MXFP4 and MXFP8). The scales require the
    following steps:
    1. Round to the closest power of 2
    2. Subtract the element-format offset so that the largest group
       values map into the quantized type's representable range
    3. Convert to biased E8M0 exponent (bias 127)

    Called when calculating qparams using observers.

    :param x: tensor of per-group max absolute values
    :param num_bits: quantized element width (4 for MXFP4, 8 for MXFP8)
    :returns scales as E8M0 exponents (uint8 after rounding)
    """
    offset = _MX_ELEM_OFFSET[num_bits]
    # Round to closest power of 2
    scale_power_2 = round_to_power_2(x)
    return 127 + torch.floor(torch.log2(scale_power_2)) - offset
