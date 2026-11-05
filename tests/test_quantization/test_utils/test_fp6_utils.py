# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.utils import calculate_qparams, cast_to_fp6
from tests.testing_utils import requires_gpu


E2M3_VALUES = torch.tensor(
    [m / 8 for m in range(8)]
    + [(1 + m / 8) * 2**e for e in range(3) for m in range(8)]
)


def _all_bf16_values() -> torch.Tensor:
    x = torch.arange(2**16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return x[~x.isnan()]


def _reference_cast(x: torch.Tensor) -> torch.Tensor:
    """Nearest E2M3 value by table search, ties to the even code, saturating."""
    a = x.float().abs().clamp(max=7.5)
    lo = (torch.bucketize(a, E2M3_VALUES, right=True) - 1).clamp(0, 31)
    hi = (lo + 1).clamp(max=31)
    d_lo, d_hi = a - E2M3_VALUES[lo], E2M3_VALUES[hi] - a
    code = torch.where((d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0)), hi, lo)
    return torch.copysign(E2M3_VALUES[code], x.float())


def test_cast_to_fp6_all_bf16_values():
    x = _all_bf16_values()
    assert torch.equal(cast_to_fp6(x).float(), _reference_cast(x))


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quantize_fp6_triton(dtype):
    # a unit scale reduces quantize to clamp + round to E2M3
    args = preset_name_to_scheme("MXFP6A16", ["Linear"]).weights
    x = _all_bf16_values()
    x = x[: x.numel() // 32 * 32].reshape(-1, 32).to(dtype).cuda()
    scale = torch.ones(x.shape[0], 1, dtype=dtype, device="cuda")

    result = quantize(x=x, scale=scale, zero_point=None, args=args)
    assert torch.equal(result.cpu().float(), _reference_cast(x.cpu()))


def test_mxfp6_scales():
    """Group maxima map into [3.875, 7.75), saturating by at most half a step."""
    args = preset_name_to_scheme("MXFP6A16", ["Linear"]).weights
    x = torch.randn(512, 64, 32) * torch.logspace(-4, 2, 512).reshape(-1, 1, 1)
    scale, _ = calculate_qparams(x.amin(-1), x.amax(-1), args)

    assert torch.equal(scale, torch.exp2(torch.log2(scale).round()))
    ratio = x.abs().amax(-1) / scale
    assert ratio.min() >= 3.875 and ratio.max() < 7.75
