# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Shared utilities for MX-format (MXFP4/MXFP8) scale compression and
decompression.

MX scales are stored as E8M0 exponents (uint8), where each value represents
a biased power-of-2 exponent with bias 127: ``scale_float = 2 ** (exp - 127)``.
"""

import torch


__all__ = ["compress_mx_scale", "decompress_mx_scale"]


def compress_mx_scale(scale: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a float scale tensor to E8M0 exponent format for MX storage.

    Extracts the power-of-2 exponent from each scale value and adds the
    E8M0 bias (127).

    :param scale: float scale tensor (e.g. from observer output)
    :param scale_dtype: target dtype for the compressed scale (typically uint8)
    :return: biased exponent tensor in the requested dtype
    """
    scale_exp = 127 + torch.floor(torch.log2(scale)).to(torch.int32)
    return scale_exp.to(scale_dtype)


def decompress_mx_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    Convert E8M0 exponent scale back to float (bfloat16).

    Reverses the E8M0 encoding by subtracting the bias and raising 2 to
    the resulting power.

    :param scale: uint8 tensor of biased E8M0 exponents
    :return: bfloat16 tensor of float scale values
    """
    scale_exp = scale.to(torch.int32) - 127
    return 2.0 ** scale_exp.to(torch.bfloat16)
