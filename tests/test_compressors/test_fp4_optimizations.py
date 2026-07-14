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

    assert torch.equal(result_current, result_reference)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pack_fp4_non_contiguous(device):
    """Test pack_fp4_to_uint8 works correctly with non-contiguous input tensors."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8

    # Create a non-contiguous tensor via transpose
    base = torch.tensor(
        [
            [0.0, -0.5, 1.0, -1.5],
            [2.0, -3.0, 4.0, -6.0],
            [0.5, -1.0, 1.5, -2.0],
            [3.0, -4.0, 6.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    x = base[:, ::2]
    assert not x.is_contiguous()

    x_contig = x.contiguous()
    result = pack_fp4_to_uint8(x)
    result_contig = pack_fp4_to_uint8(x_contig)
    expected = reference_pack_fp4_to_uint8(x_contig)

    assert torch.equal(result, result_contig)
    assert torch.equal(result, expected)
