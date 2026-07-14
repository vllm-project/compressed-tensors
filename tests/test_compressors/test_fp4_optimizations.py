# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for FP4 pack_fp4_to_uint8 optimization.

Tests that the optimized implementation produces identical results
to the original reference implementation.
"""

import pytest
import torch


FLOAT_TO_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def reference_pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation with broadcast search."""
    m, n = x.shape
    device = x.device
    if n % 2 != 0:
        raise ValueError("tensor must have an even number of columns")

    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)
    abs_x = torch.abs(x)
    abs_indices = torch.argmin(torch.abs(abs_x.unsqueeze(-1) - kE2M1), dim=-1).to(
        torch.int8
    )
    indices = abs_indices + (torch.signbit(x).to(torch.int8) << 3)
    indices = indices.reshape(-1, 2)
    packed = indices[:, 0].to(torch.uint8) | (indices[:, 1].to(torch.uint8) << 4)
    return packed.reshape(m, n // 2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "test_input",
    [
        # All positive
        torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]]),
        # All negative
        torch.tensor([[-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]]),
        # Mixed
        torch.tensor([[0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0]]),
    ],
)
def test_pack_fp4_to_uint8(device, test_input):
    """Test pack_fp4_to_uint8 matches reference implementation."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8

    x = test_input.to(dtype=torch.bfloat16, device=device)
    result_current = pack_fp4_to_uint8(x)
    result_reference = reference_pack_fp4_to_uint8(x)

    if not torch.equal(result_current, result_reference):
        # Decode packed bytes into nibble pairs for debugging
        cur = result_current.flatten().cpu().tolist()
        ref = result_reference.flatten().cpu().tolist()
        diffs = []
        for i, (c, r) in enumerate(zip(cur, ref)):
            if c != r:
                c_lo, c_hi = c & 0xF, (c >> 4) & 0xF
                r_lo, r_hi = r & 0xF, (r >> 4) & 0xF
                diffs.append(
                    f"  byte {i}: got {c} (lo={c_lo}, hi={c_hi}) "
                    f"expected {r} (lo={r_lo}, hi={r_hi})"
                )
        diff_str = "\n".join(diffs[:20])
        pytest.fail(
            f"Mismatch on {device} with input {test_input.tolist()}\n"
            f"Current:   {cur}\n"
            f"Reference: {ref}\n"
            f"Diffs:\n{diff_str}"
        )
