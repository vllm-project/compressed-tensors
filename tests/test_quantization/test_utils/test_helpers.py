# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils import (
    calculate_block_padding,
    calculate_qparams,
    compute_dynamic_scales_and_zp,
    generate_gparam,
    pad_tensor_for_block_quant,
)


@pytest.mark.parametrize(
    "keepdims,strategy,exp_shape",
    [
        (
            False,
            "tensor",
            torch.Size(
                [
                    1,
                ]
            ),
        ),
        (True, "channel", torch.Size([1, 1])),
        (True, "group", torch.Size([1, 1])),
        (
            False,
            "block",
            torch.Size(
                [
                    1,
                ]
            ),
        ),
    ],
)
def test_calculate_qparams(keepdims, strategy, exp_shape):
    value = torch.empty(5, 6)
    min_val = torch.amin(value, dim=tuple(), keepdims=keepdims)
    max_val = torch.amax(value, dim=tuple(), keepdims=keepdims)

    if strategy == QuantizationStrategy.GROUP:
        args = QuantizationArgs(strategy=strategy, group_size=2)
    elif strategy == QuantizationStrategy.BLOCK:
        args = QuantizationArgs(strategy=strategy, block_structure=[1, 3])
    else:
        args = QuantizationArgs(
            strategy=strategy,
            group_size=(2 if strategy == "group" else None),
            block_structure=([1, 3] if strategy == "block" else None),
        )
        scale, zp = calculate_qparams(min_val, max_val, args)
        assert scale.shape == exp_shape
        assert zp.shape == exp_shape


def test_fused_global_scales():
    layer = torch.nn.Linear(7, 8)
    max_tensor_value = torch.abs(layer.weight.data).max()
    # use defaults
    min_val, max_val = torch.aminmax(layer.weight)
    global_scale = generate_gparam(min_val.data, max_val.data)
    # max value should be = (448 * 6) / global_scale
    assert max_tensor_value.item() == pytest.approx(
        FP4_E2M1_DATA.max * FP8_E4M3_DATA.max / global_scale, abs=0.001
    )


@pytest.mark.parametrize(
    "shape,group_size,exp_shape",
    [
        # Only batch size =1 is supported for dynamic GROUP quantization
        ((1, 4, 8), 4, torch.Size([1, 4, 2])),
    ],
)
def test_compute_dynamic_scales_and_zp_group(shape, group_size, exp_shape):
    """
    Dynamic group quantization should reduce activations in groups, producing
    scales and zero points of shape [batch, num_groups].
    """
    value = torch.randn(*shape)
    args = QuantizationArgs(
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        dynamic=True,
    )
    scale, zp = compute_dynamic_scales_and_zp(value, args, module=torch.nn.Module())
    assert scale.shape == exp_shape
    assert zp.shape == exp_shape


# Tests for block quantization padding utilities


@pytest.mark.parametrize(
    "shape,block_structure,expected_padding",
    [
        # DeepSeek-V2-Lite intermediate_size (10944 % 128 = 64)
        ((10944, 2048), (128, 128), (64, 0)),
        # Both dimensions non-divisible
        ((100, 200), (128, 128), (28, 56)),
        # Only rows non-divisible
        ((300, 256), (128, 128), (84, 0)),
        # Only cols non-divisible
        ((256, 300), (128, 128), (0, 84)),
        # Divisible dimensions (no padding needed)
        ((256, 384), (128, 128), (0, 0)),
        # Smaller than block size
        ((100, 100), (128, 128), (28, 28)),
    ],
)
def test_calculate_block_padding(shape, block_structure, expected_padding):
    """Test that calculate_block_padding computes correct padding amounts."""
    pad_rows, pad_cols = calculate_block_padding(shape, block_structure)
    assert (pad_rows, pad_cols) == expected_padding, (
        f"For shape {shape} with block {block_structure}, "
        f"expected padding {expected_padding}, got ({pad_rows}, {pad_cols})"
    )


@pytest.mark.parametrize(
    "rows,cols,block_height,block_width",
    [
        (10944, 2048, 128, 128),  # DeepSeek-V2-Lite
        (100, 200, 128, 128),  # Both non-divisible
        (256, 256, 128, 128),  # Divisible (no padding)
        (50, 50, 128, 128),  # Smaller than block
    ],
)
def test_pad_tensor_for_block_quant(rows, cols, block_height, block_width):
    """Test that pad_tensor_for_block_quant correctly pads tensors."""
    tensor = torch.randn(rows, cols)
    block_structure = (block_height, block_width)

    padded = pad_tensor_for_block_quant(tensor, block_structure)

    # Check padded dimensions are divisible by block size
    assert (
        padded.shape[-2] % block_height == 0
    ), f"Padded rows {padded.shape[-2]} should be divisible by {block_height}"
    assert (
        padded.shape[-1] % block_width == 0
    ), f"Padded cols {padded.shape[-1]} should be divisible by {block_width}"

    # Check that original values are preserved
    assert torch.equal(
        padded[:rows, :cols], tensor
    ), "Original values should be preserved in padded tensor"

    # Check that padding is zeros
    if padded.shape[-2] > rows:
        assert torch.all(padded[rows:, :] == 0), "Row padding should be zeros"
    if padded.shape[-1] > cols:
        assert torch.all(padded[:, cols:] == 0), "Column padding should be zeros"
