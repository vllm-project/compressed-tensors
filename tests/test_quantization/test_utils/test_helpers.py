# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from compressed_tensors.quantization import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import (
    calculate_block_padding,
    calculate_qparams,
    compute_dynamic_scales_and_zp,
    generate_gparam,
    maybe_pad_tensor_for_block_quant,
)


@pytest.mark.parametrize("symmetric", [True, False])
def test_calculate_qparams_meta_device(symmetric):
    # ensure qparams can be computed for tensors on the meta device
    # (e.g. no python float conversions that would fail on meta tensors)
    min_val = torch.zeros(1, device="meta")
    max_val = torch.zeros(1, device="meta")
    args = QuantizationArgs(strategy="tensor", symmetric=symmetric)

    scale, zp = calculate_qparams(min_val, max_val, args)

    assert scale.device.type == "meta"
    assert zp.device.type == "meta"


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


@pytest.mark.regression
@pytest.mark.parametrize("shape", [(8,), (2, 8), (1, 2, 8), (2, 1, 2, 8)])
@pytest.mark.parametrize("symmetric", [True, False])
def test_compute_dynamic_scales_and_zp_token(shape, symmetric):
    """Each token gets its own qparams regardless of activation rank."""
    value = torch.arange(1, math.prod(shape) + 1, dtype=torch.float32).reshape(shape)
    # Give each token a different range, including negative values.
    value[..., ::2] *= -1
    args = QuantizationArgs(strategy="token", dynamic=True, symmetric=symmetric)

    scale, zp = compute_dynamic_scales_and_zp(value, args, module=torch.nn.Module())
    expected = [
        calculate_qparams(*torch.aminmax(row), args) for row in value.view(-1, 8)
    ]
    expected_shape = (*shape[:-1], 1)

    assert scale.shape == expected_shape
    assert zp.shape == expected_shape
    torch.testing.assert_close(
        scale, torch.cat([s for s, _ in expected]).reshape(expected_shape)
    )
    torch.testing.assert_close(
        zp, torch.cat([z for _, z in expected]).reshape(expected_shape)
    )


@pytest.mark.regression
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize(
    "value",
    [
        [[40.0, -30.0, 20.0, -10.0], [2.7, -1.6, 0.5, -0.4]],
        [[0.0] * 4, [1.0] * 4, [-2.0] * 4],
    ],
)
def test_dynamic_token_quantization_flattened_activations(value, symmetric):
    """Flattening batch dimensions preserves per-token quantized values."""
    value = torch.tensor(value)
    args = QuantizationArgs(strategy="token", dynamic=True, symmetric=symmetric)
    expected = []
    for row in value:
        scale, zp = calculate_qparams(*torch.aminmax(row), args)
        expected.append(fake_quantize(row, scale, zp, args))
    expected = torch.stack(expected)

    for activations in (value, value.unsqueeze(0), value.unsqueeze(0).unsqueeze(0)):
        scale, zp = compute_dynamic_scales_and_zp(
            activations, args, module=torch.nn.Module()
        )
        output = fake_quantize(activations, scale, zp, args).reshape_as(value)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected)


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
def test_maybe_pad_tensor_for_block_quant(rows, cols, block_height, block_width):
    """Test that maybe_pad_tensor_for_block_quant correctly pads tensors."""
    tensor = torch.randn(rows, cols)
    block_structure = (block_height, block_width)

    padded = maybe_pad_tensor_for_block_quant(tensor, block_structure)

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
