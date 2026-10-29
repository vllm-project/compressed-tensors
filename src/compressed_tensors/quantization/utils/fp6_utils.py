# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.utils.triton import tl, triton


__all__ = ["_round_to_fp6", "cast_to_fp6"]

# FP6 E2M3 has a 3-bit mantissa and exponent bias 1, so a magnitude with float
# exponent e is a multiple of 2 ** (max(e, 0) - 3): steps of 0.125 below 2,
# 0.25 in [2, 4) and 0.5 in [4, 7.5].


@triton.jit
def _round_to_fp6(x, use_intel_libdevice: tl.constexpr):
    """
    Round float values in [-7.5, 7.5] to the nearest FP6 E2M3 value, ties to even.
    """
    x = x.to(tl.float32)
    abs_x = tl.abs(x)
    exp = tl.maximum(((abs_x.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127, 0)
    step = ((exp - 3 + 127) << 23).to(tl.float32, bitcast=True)
    if use_intel_libdevice:
        rounded = tl.extra.intel.libdevice.rint(abs_x / step) * step
    else:
        rounded = tl.extra.cuda.libdevice.rint(abs_x / step) * step
    return tl.where(x < 0.0, -rounded, rounded)


def cast_to_fp6(x: torch.Tensor) -> torch.Tensor:
    """
    Round float values to the nearest FP6 E2M3 value, ties to even, saturating
    at +/-7.5.

    :param x: input tensor
    :return: rounded tensor with the same dtype as the input
    """
    abs_x = x.abs().float()
    exp = (((abs_x.view(torch.int32) >> 23) & 0xFF) - 127).clamp_(min=0)
    step = torch.exp2(exp - 3.0)
    rounded = torch.round(abs_x / step).mul_(step).clamp_(max=7.5)
    return torch.copysign(rounded, x.float()).to(x.dtype)
