# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from compressed_tensors.quantization.lifecycle.forward import (
    _process_quantization,
    fake_quantize,
    forward_quantize,
    set_forward_quantized,
)
from compressed_tensors.quantization.lifecycle.forward_helpers import (
    _dequantize,
    _quantize,
    _quantize_dequantize,
)
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.utils.helpers import calculate_range
from torch.nn import Embedding, Linear


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.tensor([index // group_size for index in range(columns)])[perm]


def test_set_forward_quantized():
    layer = Linear(4, 4)
    func_forward = layer.forward.__func__

    # check that the forward call is overwritten
    set_forward_quantized(layer)
    assert not func_forward == layer.forward.__func__


def test_set_forward_quantized_embedding():
    """Test that set_forward_quantized works with Embedding modules"""
    embedding = Embedding(num_embeddings=10, embedding_dim=4)
    func_forward = embedding.forward.__func__

    # check that the forward call is overwritten
    set_forward_quantized(embedding)
    assert not func_forward == embedding.forward.__func__


def test_set_forward_quantized_embedding_no_quantization():
    """
    Test forward pass of Embedding when quantization is disabled or
    scheme is not set
    """
    embedding = Embedding(num_embeddings=10, embedding_dim=4)
    set_forward_quantized(embedding)

    input_indices = torch.tensor([0, 1, 2, 3])
    expected_output = torch.nn.functional.embedding(input_indices, embedding.weight)

    # Without quantization scheme, should behave like normal embedding
    output = embedding(input_indices)
    assert torch.allclose(output, expected_output)


def test_set_forward_quantized_embedding_with_weight_quantization(
    mock_per_tensor_calibration, create_quantization_scheme
):
    """Test forward pass with weight quantization on Embedding module"""
    num_bits = 8
    embedding = Embedding(num_embeddings=10, embedding_dim=4)
    embedding.weight.data *= 10

    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )

    # initialize_module_for_quantization calls set_forward_quantized
    initialize_module_for_quantization(embedding, quantization_scheme)
    embedding.quantization_status = QuantizationStatus.CALIBRATION

    # Calibrate weights
    mock_per_tensor_calibration(embedding, "weight", value=embedding.weight.data)

    # Forward pass should quantize weights
    input_indices = torch.tensor([0, 1, 2, 3])
    output = embedding(input_indices)
    assert output.shape == (4, 4)

    # Output should be different from unquantized forward
    unquantized_output = torch.nn.functional.embedding(input_indices, embedding.weight)
    assert not torch.allclose(output, unquantized_output, atol=1e-3)


def test_set_forward_quantized_no_quantization():
    """Test forward pass when quantization is disabled or scheme is not set"""
    layer = Linear(4, 4)
    set_forward_quantized(layer)

    input_tensor = torch.randn(2, 4)
    expected_output = torch.nn.functional.linear(input_tensor, layer.weight, layer.bias)

    # Without quantization scheme, should behave like normal linear
    output = layer(input_tensor)
    assert torch.allclose(output, expected_output)


def test_set_forward_quantized_disabled():
    """Test forward pass when quantization_enabled is False"""
    layer = Linear(4, 4)
    set_forward_quantized(layer)

    # Set up quantization but disable it
    layer.quantization_enabled = False
    layer.quantization_scheme = torch.nn.Module()  # dummy scheme
    layer.quantization_status = QuantizationStatus.INITIALIZED

    input_tensor = torch.randn(2, 4)
    expected_output = torch.nn.functional.linear(input_tensor, layer.weight, layer.bias)

    # With quantization disabled, should behave like normal linear
    output = layer(input_tensor)
    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize(
    "quantization_status",
    [
        QuantizationStatus.INITIALIZED,
        QuantizationStatus.CALIBRATION,
        QuantizationStatus.FROZEN,
    ],
)
def test_set_forward_quantized_with_input_activations(
    mock_per_tensor_calibration, create_quantization_scheme, quantization_status
):
    """Test forward pass with input activation quantization"""
    num_bits = 8
    layer = Linear(4, 4)
    layer.weight.data *= 10

    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )

    # initialize_module_for_quantization calls set_forward_quantized
    initialize_module_for_quantization(layer, quantization_scheme)
    layer.quantization_status = quantization_status

    # Calibrate input activations
    input_tensor = torch.randn(2, 4)
    mock_per_tensor_calibration(layer, "input", value=input_tensor)

    # Forward pass should quantize inputs
    output = layer(input_tensor)
    assert output.shape == (2, 4)
    # Output should be different from unquantized forward
    unquantized_output = torch.nn.functional.linear(
        input_tensor, layer.weight, layer.bias
    )
    assert not torch.allclose(output, unquantized_output, atol=1e-3)


@pytest.mark.parametrize(
    "quantization_status",
    [
        QuantizationStatus.INITIALIZED,
        QuantizationStatus.CALIBRATION,
    ],
)
def test_set_forward_quantized_with_weight_quantization(
    mock_per_tensor_calibration, create_quantization_scheme, quantization_status
):
    """Test forward pass with weight quantization (non-FROZEN status)"""
    num_bits = 8
    layer = Linear(4, 4)
    layer.weight.data *= 10

    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )

    # initialize_module_for_quantization calls set_forward_quantized
    initialize_module_for_quantization(layer, quantization_scheme)
    layer.quantization_status = quantization_status

    # Calibrate weights
    mock_per_tensor_calibration(layer, "weight", value=layer.weight.data)

    # Forward pass should quantize weights
    input_tensor = torch.randn(2, 4)
    output = layer(input_tensor)
    assert output.shape == (2, 4)


def test_set_forward_quantized_compressed_status(
    mock_per_tensor_calibration, create_quantization_scheme
):
    """Test that weight quantization is skipped when status is FROZEN"""
    num_bits = 8
    layer = Linear(4, 4)
    layer.weight.data *= 10

    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )

    # initialize_module_for_quantization calls set_forward_quantized
    initialize_module_for_quantization(layer, quantization_scheme)
    layer.quantization_status = QuantizationStatus.COMPRESSED

    # Calibrate weights
    mock_per_tensor_calibration(layer, "weight", value=layer.weight.data)

    # Forward pass should NOT quantize weights due to FROZEN status
    input_tensor = torch.randn(2, 4)
    output = layer(input_tensor)
    expected_output = torch.nn.functional.linear(input_tensor, layer.weight, layer.bias)
    assert torch.allclose(output, expected_output)


def test_set_forward_quantized_with_output_activations(
    mock_per_tensor_calibration, create_quantization_scheme
):
    """Test forward pass with output activation quantization"""
    num_bits = 8
    layer = Linear(4, 4)
    layer.weight.data *= 10

    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        output_activations=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )

    # initialize_module_for_quantization calls set_forward_quantized
    initialize_module_for_quantization(layer, quantization_scheme)
    layer.quantization_status = QuantizationStatus.CALIBRATION

    # Need to calibrate output activations
    input_tensor = torch.randn(2, 4)
    output_sample = torch.nn.functional.linear(input_tensor, layer.weight, layer.bias)
    mock_per_tensor_calibration(layer, "output", value=output_sample)

    # Forward pass should quantize outputs
    output = layer(input_tensor)
    assert output.shape == (2, 4)


def test_set_forward_quantized_full_quantization(
    mock_per_tensor_calibration, create_quantization_scheme
):
    """Test forward pass with input, weight, and output quantization enabled"""
    num_bits = 8
    layer = Linear(4, 4)
    layer.weight.data *= 10

    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=True),
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        output_activations=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )

    # initialize_module_for_quantization calls set_forward_quantized
    initialize_module_for_quantization(layer, quantization_scheme)
    layer.quantization_status = QuantizationStatus.CALIBRATION

    # Calibrate all components
    input_tensor = torch.randn(2, 4)
    mock_per_tensor_calibration(layer, "weight", value=layer.weight.data)
    mock_per_tensor_calibration(layer, "input", value=input_tensor)
    output_sample = torch.nn.functional.linear(input_tensor, layer.weight, layer.bias)
    mock_per_tensor_calibration(layer, "output", value=output_sample)

    # Forward pass should quantize all components
    output = layer(input_tensor)
    assert output.shape == (2, 4)
    # Should be significantly different from unquantized
    unquantized_output = torch.nn.functional.linear(
        input_tensor, layer.weight, layer.bias
    )
    assert not torch.allclose(output, unquantized_output, atol=1e-2)


@pytest.mark.parametrize("quantization_status", ["initialized", "calibration"])
def test_forward_quantize(
    mock_per_tensor_calibration, create_quantization_scheme, quantization_status
):
    num_bits = 8
    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=True),
        input_activations=QuantizationArgs(num_bits=num_bits, symmetric=True),
    )
    quantization_args = QuantizationArgs(num_bits=num_bits, symmetric=True)
    layer = Linear(4, 4)
    layer.weight.data *= 100

    dummy_tensor = torch.randn(8, 4)  # (num_tokens, num_features)
    layer.quantization_status = QuantizationStatus(quantization_status)

    # only calibration updates the scale and zero-point
    if layer.quantization_status == QuantizationStatus.INITIALIZED:
        # Init zp and scales
        initialize_module_for_quantization(layer, quantization_scheme)
        # mock weight calibration
        mock_per_tensor_calibration(layer, "weight", value=layer.weight.data)
        # call quant/dequant on weights
        out = forward_quantize(layer, layer.weight, "weight", quantization_args)
        assert torch.allclose(out, layer.weight.data, atol=0.2)
    elif layer.quantization_status == QuantizationStatus.CALIBRATION:
        # init zp/scales
        initialize_module_for_quantization(layer, quantization_scheme)
        # run weight and input calibration
        mock_per_tensor_calibration(layer, "weight", value=layer.weight.data)
        mock_per_tensor_calibration(layer, "input", value=dummy_tensor)
        # call quant/dequant on inputs
        out = forward_quantize(layer, dummy_tensor, "input", quantization_args)
        assert torch.allclose(out, dummy_tensor, atol=0.2)


@pytest.mark.parametrize(
    "num_bits,type,strategy,group_size,scale,zero_point,g_idx,global_scale",
    [
        (
            4,
            "int",
            QuantizationStrategy.TENSOR,
            None,
            torch.rand((1,)) * 0.01,
            torch.zeros((1,)),
            None,
            None,
        ),
        (
            4,
            "int",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8)) * 0.01,
            torch.zeros((512, 8)),
            None,
            None,
        ),
        (
            4,
            "int",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8)) * 0.01,
            torch.zeros((512, 8)),
            make_dummy_g_idx(1024, 128),
            None,
        ),
        (
            8,
            "float",
            QuantizationStrategy.TENSOR,
            None,
            torch.rand((1,)) * 0.01,
            torch.zeros((1,)),
            None,
            None,
        ),
        (
            8,
            "float",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8)) * 0.01,
            torch.zeros((512, 8)),
            None,
            None,
        ),
        (
            8,
            "float",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8)) * 0.01,
            torch.zeros((512, 8)),
            make_dummy_g_idx(1024, 128),
            None,
        ),
        (
            8,
            "int",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8)) * 0.01,
            torch.zeros((512, 8)),
            None,
            None,
        ),
        (
            8,
            "int",
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8)) * 0.01,
            torch.zeros((512, 8)),
            make_dummy_g_idx(1024, 128),
            None,
        ),
    ],
)
def test_fake_quantize_2d(
    num_bits, type, strategy, group_size, scale, zero_point, g_idx, global_scale
):
    args = QuantizationArgs(
        num_bits=num_bits, type=type, strategy=strategy, group_size=group_size
    )

    x = torch.rand((512, 1024))
    fake_quantize(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        g_idx=g_idx,
        global_scale=global_scale,
    )  # note that reconstruction loss is bad for uncalibrated scales


def test_process_quantization_block_static():
    """
    Static block quantization (QuantizationStrategy.BLOCK) should split a 2D tensor
    into blocks, quantize each block, and reassemble without changing shape.
    """
    rows, cols = 8, 8
    bh, bw = 2, 4
    x = torch.randn(rows, cols)
    args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[bh, bw],
    )
    num_rb = math.ceil(rows / bh)
    num_cb = math.ceil(cols / bw)
    scale = torch.rand(num_rb, num_cb) + 0.1
    zp = torch.zeros_like(scale)
    q_min, q_max = calculate_range(args, x.device)
    out = _process_quantization(
        x=x,
        scale=scale,
        zero_point=zp,
        args=args,
        do_quantize=True,
        do_dequantize=False,
        dtype=None,
        global_scale=None,
    )
    assert out.shape == x.shape
    # full fake-quantize roundtrip
    out2 = _process_quantization(
        x=x,
        scale=scale,
        zero_point=zp,
        args=args,
        do_quantize=True,
        do_dequantize=True,
        dtype=None,
        global_scale=None,
    )
    assert out2.shape == x.shape


@pytest.mark.parametrize(
    "rows,cols,block_height,block_width",
    [
        (4544, 768, 128, 128),  # Falcon-7B dimensions: 4544 = 64*71
        (100, 200, 128, 128),  # Both dimensions not divisible
        (256, 300, 128, 128),  # Only cols not divisible
        (300, 256, 128, 128),  # Only rows not divisible
        (127, 127, 128, 128),  # Both dimensions smaller than block size
        (1, 1, 128, 128),  # Minimal tensor
    ],
)
def test_process_quantization_block_non_divisible(
    rows, cols, block_height, block_width
):
    """
    Block quantization should handle tensor dimensions that are not divisible
    by the block size by padding internally.
    """
    x = torch.randn(rows, cols)
    args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[block_height, block_width],
    )
    # Calculate number of blocks (with ceiling division for padding)
    num_rb = math.ceil(rows / block_height)
    num_cb = math.ceil(cols / block_width)
    scale = torch.rand(num_rb, num_cb) + 0.1
    zp = torch.zeros_like(scale)

    # Should NOT raise ValueError anymore
    out = _process_quantization(
        x=x,
        scale=scale,
        zero_point=zp,
        args=args,
        do_quantize=True,
        do_dequantize=False,
        dtype=None,
        global_scale=None,
    )
    # Output shape should match original input shape
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    # Full fake-quantize roundtrip
    out2 = _process_quantization(
        x=x,
        scale=scale,
        zero_point=zp,
        args=args,
        do_quantize=True,
        do_dequantize=True,
        dtype=None,
        global_scale=None,
    )
    assert out2.shape == x.shape, f"Expected {x.shape}, got {out2.shape}"


@pytest.mark.parametrize(
    "rows,cols,block_height,block_width",
    [
        (100, 200, 128, 128),  # Both dimensions not divisible
        (256, 300, 128, 128),  # Only cols not divisible
        (300, 256, 128, 128),  # Only rows not divisible
        (127, 127, 128, 128),  # Both dimensions smaller than block size
    ],
)
def test_process_quantization_block_non_divisible_values(
    rows, cols, block_height, block_width
):
    """
    Verify that block quantization with non-divisible dimensions produces
    correct values. Using uniform input (ones) with scale=1.0 should result
    in zero quantization loss.
    """
    # Use uniform values - quantization with scale=1.0 should be lossless
    x = torch.ones(rows, cols)
    args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[block_height, block_width],
    )
    num_rb = math.ceil(rows / block_height)
    num_cb = math.ceil(cols / block_width)
    # Use scale=1.0 for lossless quantization of values within FP8 range
    scale = torch.ones(num_rb, num_cb)
    zp = torch.zeros_like(scale)

    # Full fake-quantize roundtrip should preserve values exactly
    out = _process_quantization(
        x=x,
        scale=scale,
        zero_point=zp,
        args=args,
        do_quantize=True,
        do_dequantize=True,
        dtype=None,
        global_scale=None,
    )

    # Values should match input (no quantization loss for uniform values)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    assert torch.allclose(
        out, x, atol=1e-6
    ), f"Values mismatch: expected all ones, got min={out.min()}, max={out.max()}"

    # Test with a different uniform value
    x_val = torch.full((rows, cols), 0.5)
    out_val = _process_quantization(
        x=x_val,
        scale=scale,
        zero_point=zp,
        args=args,
        do_quantize=True,
        do_dequantize=True,
        dtype=None,
        global_scale=None,
    )
    assert torch.allclose(
        out_val, x_val, atol=1e-6
    ), f"Values mismatch for 0.5: got min={out_val.min()}, max={out_val.max()}"


@pytest.mark.parametrize(
    "num_bits,type,symmetric,global_scale",
    [
        (8, "int", True, None),
        (8, "int", False, None),
        (4, "int", True, None),
        (8, "float", True, None),
        (8, "float", True, torch.tensor([2.0])),
        (8, "int", False, torch.tensor([2.0])),
    ],
)
def test_quantize_dequantize_matches_sequential(
    num_bits, type, symmetric, global_scale
):
    """Verify that the fused _quantize_dequantize produces identical output
    to calling _quantize then _dequantize sequentially."""
    args = QuantizationArgs(
        num_bits=num_bits,
        type=type,
        symmetric=symmetric,
        strategy=QuantizationStrategy.TENSOR,
    )
    q_min, q_max = calculate_range(args, torch.device("cpu"))

    x = torch.randn(512, 1024)
    scale = torch.rand(1) * 0.01 + 0.001
    zero_point = None if symmetric else torch.tensor([3.0])

    # sequential: quantize then dequantize
    q = _quantize(
        x=x,
        scale=scale,
        zero_point=zero_point,
        q_min=q_min,
        q_max=q_max,
        args=args,
        global_scale=global_scale,
    )
    sequential_out = _dequantize(
        x_q=q,
        scale=scale,
        zero_point=zero_point,
        global_scale=global_scale,
    )

    # fused
    fused_out = _quantize_dequantize(
        x=x,
        scale=scale,
        zero_point=zero_point,
        q_min=q_min,
        q_max=q_max,
        args=args,
        global_scale=global_scale,
    )

    assert torch.equal(
        sequential_out, fused_out
    ), f"Mismatch: max diff = {(sequential_out - fused_out).abs().max().item()}"
