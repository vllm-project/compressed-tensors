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
    _is_fp8_supported,
    _needs_fp8,
    _quantize,
    _quantize_dequantize,
    adapt_scale_and_zp_for_triton,
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


@pytest.mark.parametrize("device", ["cpu", "cuda", "meta"])
@pytest.mark.parametrize(
    "num_bits,type,symmetric,global_scale,group_size",
    [
        # Tensor-level quantization (group_size=None)
        (8, "int", True, None, None),
        (8, "int", False, None, None),
        (4, "int", True, None, None),
        (4, "float", True, None, None),  # FP4
        (8, "float", True, None, None),
        (8, "float", True, torch.tensor([2.0]), None),
        (8, "int", False, torch.tensor([2.0]), None),
        # Group quantization
        (8, "int", True, None, 128),
        (8, "int", False, None, 128),
        (4, "int", True, None, 128),
        (4, "float", True, None, 128),  # FP4
        (8, "float", True, None, 128),
        (8, "float", True, torch.tensor([2.0]), 128),
        (8, "int", False, torch.tensor([2.0]), 128),
        (8, "int", True, None, 64),
        (8, "int", False, None, 256),
    ],
)
def test_quantize_dequantize_matches_sequential(
    num_bits, type, symmetric, global_scale, group_size, device
):
    """Verify that the fused _quantize_dequantize produces identical output
    to calling _quantize then _dequantize sequentially."""
    if device == "cuda" and not torch.accelerator.is_available():
        pytest.skip("CUDA not available")
    if device == "cpu" and type == "float" and num_bits == 4:
        pytest.skip("FP4 on CPU is slow, only test on CUDA")

    num_rows = 512
    num_cols = 1024

    if group_size is None:
        strategy = QuantizationStrategy.TENSOR
        args = QuantizationArgs(
            num_bits=num_bits,
            type=type,
            symmetric=symmetric,
            strategy=strategy,
        )
        x = torch.randn(num_rows, num_cols, device=device)
        scale = (torch.rand(1) * 0.01 + 0.001).to(device)
        zero_point = None if symmetric else torch.tensor([3.0], device=device)
    else:
        strategy = QuantizationStrategy.GROUP
        num_groups = num_cols // group_size
        args = QuantizationArgs(
            num_bits=num_bits,
            type=type,
            symmetric=symmetric,
            strategy=strategy,
            group_size=group_size,
        )
        x = torch.randn(num_rows, num_groups, group_size, device=device)
        scale = torch.rand(num_rows, num_groups, 1, device=device) * 0.01 + 0.001
        zero_point = (
            None
            if symmetric
            else torch.zeros(num_rows, num_groups, 1, device=device) + 3.0
        )

    q_min, q_max = calculate_range(args, torch.device(device))
    if global_scale is not None:
        global_scale = global_scale.to(device)

    # Determine if Triton should be used
    is_gpu = x.is_cuda or x.is_xpu
    is_fp8 = _needs_fp8(x, scale, zero_point, global_scale, args=args)
    fp8_hw_ok = _is_fp8_supported(x.device) if is_fp8 else True
    do_triton = is_gpu and (not is_fp8 or fp8_hw_ok)

    # Keep original scale/zp for dequantize
    scale_ground_dequant = scale.clone()
    zero_point_dequant = zero_point.clone() if zero_point is not None else None

    # Adapt scale/zp for Triton if needed (for _quantize only)
    scale_quant = scale.clone()
    zero_point_quant = zero_point.clone() if zero_point is not None else None
    if do_triton:
        num_rows = x.shape[0]
        scale_quant, zero_point_quant = adapt_scale_and_zp_for_triton(
            scale_quant, zero_point_quant, num_rows
        )

    # sequential: quantize then dequantize
    q = _quantize(
        x=x,
        scale=scale_quant,
        zero_point=zero_point_quant,
        q_min=q_min,
        q_max=q_max,
        args=args,
        global_scale=global_scale,
        do_triton=do_triton,
    )
    sequential_out = _dequantize(
        x_q=q,
        scale=scale_ground_dequant,
        zero_point=zero_point_dequant,
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

    if device == "meta":
        return  # Meta tensors have no data; can only verify code doesn't crash

    if type == "int":
        atol, rtol = 1.0, 0  # allow +/-1 due to rounding corner cases
    else:
        atol, rtol = 1e-5, 0.15

    assert torch.allclose(sequential_out, fused_out, atol=atol, rtol=rtol), (
        f"Mismatch: max diff = {(sequential_out - fused_out).abs().max().item()}, "
        f"atol={atol}, rtol={rtol}"
    )

@pytest.mark.parametrize(
    "num_bits,type,symmetric,global_scale,strategy,group_size",
    [
        # Per-tensor (scalar scale) - uses fast path
        (8, "int", True, None, QuantizationStrategy.TENSOR, None),
        (8, "int", False, None, QuantizationStrategy.TENSOR, None),
        (4, "int", True, None, QuantizationStrategy.TENSOR, None),
        (4, "float", True, None, QuantizationStrategy.TENSOR, None),  # FP4
        (8, "float", True, None, QuantizationStrategy.TENSOR, None),
        (8, "float", True, torch.tensor([2.0]), QuantizationStrategy.TENSOR, None),
        (8, "int", False, torch.tensor([2.0]), QuantizationStrategy.TENSOR, None),
        # Per-channel (one scale per row) - uses grouped path
        (8, "int", True, None, QuantizationStrategy.CHANNEL, None),
        (8, "int", False, None, QuantizationStrategy.CHANNEL, None),
        (4, "int", True, None, QuantizationStrategy.CHANNEL, None),
        # Per-group (multiple scales per row) - uses grouped path
        (8, "int", True, None, QuantizationStrategy.GROUP, 128),
        (8, "int", False, None, QuantizationStrategy.GROUP, 128),
        (4, "int", True, None, QuantizationStrategy.GROUP, 128),
        (4, "int", True, None, QuantizationStrategy.GROUP, 64),
        # Per-group with global_scale
        (8, "int", True, torch.tensor([2.0]), QuantizationStrategy.GROUP, 128),
    ],
)
def test_dequantize_triton_matches_cpu(
    num_bits, type, symmetric, global_scale, strategy, group_size
):
    """Verify Triton _dequantize on GPU matches CPU implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    args = QuantizationArgs(
        num_bits=num_bits,
        type=type,
        symmetric=symmetric,
        strategy=strategy,
        group_size=group_size,
    )

    # Create input on CPU first - use dimensions divisible by common group sizes
    num_rows, num_cols = 512, 1024
    x_cpu = torch.randn(num_rows, num_cols)

    # Create scale based on strategy
    if strategy == QuantizationStrategy.TENSOR:
        scale_cpu = torch.rand(1) * 0.01 + 0.001
        zero_point_cpu = None if symmetric else torch.tensor([3.0])
    elif strategy == QuantizationStrategy.CHANNEL:
        # One scale per row (channel)
        scale_cpu = torch.rand(num_rows, 1) * 0.01 + 0.001
        zero_point_cpu = None if symmetric else torch.randint(1, 5, (num_rows, 1)).float()
    elif strategy == QuantizationStrategy.GROUP:
        # One scale per group: shape (num_rows, num_cols // group_size)
        num_groups = num_cols // group_size
        scale_cpu = torch.rand(num_rows, num_groups) * 0.01 + 0.001
        zero_point_cpu = (
            None if symmetric else torch.randint(1, 5, (num_rows, num_groups)).float()
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    global_scale_cpu = global_scale.clone() if global_scale is not None else None

    q_min_cpu, q_max_cpu = calculate_range(args, torch.device("cpu"))

    # Compute effective scale for quantization
    effective_scale = scale_cpu
    if global_scale_cpu is not None:
        effective_scale = scale_cpu / global_scale_cpu

    # Quantize first to get valid quantized values
    # For group/channel, we need to broadcast scale properly
    if strategy == QuantizationStrategy.GROUP:
        # Expand scale to match input shape for quantization
        scale_expanded = effective_scale.repeat_interleave(group_size, dim=1)
        x_q_cpu = torch.clamp(torch.round(x_cpu / scale_expanded), q_min_cpu, q_max_cpu)
        if zero_point_cpu is not None:
            zp_expanded = zero_point_cpu.repeat_interleave(group_size, dim=1)
            x_q_cpu = x_q_cpu + zp_expanded
    elif strategy == QuantizationStrategy.CHANNEL:
        # Scale broadcasts along columns
        x_q_cpu = torch.clamp(torch.round(x_cpu / effective_scale), q_min_cpu, q_max_cpu)
        if zero_point_cpu is not None:
            x_q_cpu = x_q_cpu + zero_point_cpu
    else:
        # Tensor strategy - scalar scale
        x_q_cpu = torch.clamp(torch.round(x_cpu / effective_scale), q_min_cpu, q_max_cpu)
        if zero_point_cpu is not None:
            x_q_cpu = x_q_cpu + zero_point_cpu

    # For GROUP strategy, use broadcasting like real workloads (_process_group):
    # - Reshape x_q to 3D: (num_rows, num_groups, group_size)
    # - Unsqueeze scale to (num_rows, num_groups, 1) for broadcasting
    if strategy == QuantizationStrategy.GROUP:
        num_groups = num_cols // group_size
        x_q_3d = x_q_cpu.reshape(num_rows, num_groups, group_size)
        # scale.unsqueeze(-1) matches how _process_group prepares scale
        scale_3d = scale_cpu.unsqueeze(-1)  # (num_rows, num_groups, 1)
        zp_3d = zero_point_cpu.unsqueeze(-1) if zero_point_cpu is not None else None

        # Manual CPU reference using broadcasting (mirrors real workload)
        effective_scale = scale_3d
        if global_scale_cpu is not None:
            effective_scale = scale_3d / global_scale_cpu

        cpu_out = x_q_3d.to(scale_cpu.dtype)
        if zp_3d is not None:
            cpu_out = cpu_out - zp_3d.to(scale_cpu.dtype)
        cpu_out = cpu_out * effective_scale  # Broadcasting across group_size dim
    else:
        # TENSOR/CHANNEL: CPU fallback in _dequantize handles these correctly
        cpu_out = _dequantize(
            x_q=x_q_cpu,
            scale=scale_cpu,
            zero_point=zero_point_cpu,
            global_scale=global_scale_cpu,
        )

    # Copy to CUDA and run Triton path
    # For GROUP strategy, reshape to 3D: (num_rows, num_groups, group_size)
    # This is what _dequantize_grouped expects for proper group handling
    if strategy == QuantizationStrategy.GROUP:
        x_q_cuda = x_q_3d.cuda()
        scale_cuda = scale_cpu.cuda()  # Shape: (num_rows, num_groups)
        zero_point_cuda = zero_point_cpu.cuda() if zero_point_cpu is not None else None
    else:
        x_q_cuda = x_q_cpu.cuda()
        scale_cuda = scale_cpu.cuda()
        zero_point_cuda = zero_point_cpu.cuda() if zero_point_cpu is not None else None

    global_scale_cuda = (
        global_scale_cpu.cuda() if global_scale_cpu is not None else None
    )

    cuda_out = _dequantize(
        x_q=x_q_cuda,
        scale=scale_cuda,
        zero_point=zero_point_cuda,
        global_scale=global_scale_cuda,
    )

    assert torch.allclose(
        cpu_out, cuda_out.cpu(), rtol=1e-5, atol=1e-5
    ), f"Mismatch: max diff = {(cpu_out - cuda_out.cpu()).abs().max().item()}"


@pytest.mark.parametrize(
    "num_bits,type,symmetric,global_scale,strategy,group_size",
    [
        # Per-tensor (scalar scale) - uses fast scalar kernel
        (8, "int", True, None, QuantizationStrategy.TENSOR, None),
        (8, "int", False, None, QuantizationStrategy.TENSOR, None),
        (4, "int", True, None, QuantizationStrategy.TENSOR, None),
        (4, "float", True, None, QuantizationStrategy.TENSOR, None),  # FP4
        (8, "float", True, None, QuantizationStrategy.TENSOR, None),
        (8, "float", True, torch.tensor([2.0]), QuantizationStrategy.TENSOR, None),
        (8, "int", False, torch.tensor([2.0]), QuantizationStrategy.TENSOR, None),
        # Per-channel (one scale per row) - uses grouped kernel
        (8, "int", True, None, QuantizationStrategy.CHANNEL, None),
        (8, "int", False, None, QuantizationStrategy.CHANNEL, None),
        (4, "int", True, None, QuantizationStrategy.CHANNEL, None),
        # Per-group (multiple scales per row) - uses grouped kernel
        (8, "int", True, None, QuantizationStrategy.GROUP, 128),
        (8, "int", False, None, QuantizationStrategy.GROUP, 128),
        (4, "int", True, None, QuantizationStrategy.GROUP, 128),
        (4, "int", True, None, QuantizationStrategy.GROUP, 64),
        # Per-group with global_scale
        (8, "int", True, torch.tensor([2.0]), QuantizationStrategy.GROUP, 128),
    ],
)
def test_quantize_dequantize_triton_matches_cpu(
    num_bits, type, symmetric, global_scale, strategy, group_size
):
    """Verify Triton _quantize_dequantize on GPU matches CPU implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from compressed_tensors.quantization.lifecycle.forward_helpers import (
        _quantize_dequantize,
    )

    args = QuantizationArgs(
        num_bits=num_bits,
        type=type,
        symmetric=symmetric,
        strategy=strategy,
        group_size=group_size,
    )

    # Create input on CPU first - use dimensions divisible by common group sizes
    num_rows, num_cols = 512, 1024
    x_cpu = torch.randn(num_rows, num_cols)

    # Create scale based on strategy
    if strategy == QuantizationStrategy.TENSOR:
        scale_cpu = torch.rand(1) * 0.1 + 0.01
        zero_point_cpu = None if symmetric else torch.tensor([3.0])
    elif strategy == QuantizationStrategy.CHANNEL:
        # One scale per row (channel)
        scale_cpu = torch.rand(num_rows, 1) * 0.1 + 0.01
        zero_point_cpu = None if symmetric else torch.randint(1, 5, (num_rows, 1)).float()
    elif strategy == QuantizationStrategy.GROUP:
        # One scale per group: shape (num_rows, num_cols // group_size)
        num_groups = num_cols // group_size
        scale_cpu = torch.rand(num_rows, num_groups) * 0.1 + 0.01
        zero_point_cpu = (
            None if symmetric else torch.randint(1, 5, (num_rows, num_groups)).float()
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    global_scale_cpu = global_scale.clone() if global_scale is not None else None

    q_min_cpu, q_max_cpu = calculate_range(args, torch.device("cpu"))

    # For GROUP strategy, reshape to 3D as _process_group would do
    if strategy == QuantizationStrategy.GROUP:
        num_groups = num_cols // group_size
        x_3d = x_cpu.reshape(num_rows, num_groups, group_size)
        scale_3d = scale_cpu.unsqueeze(-1)  # (num_rows, num_groups, 1)
        zp_3d = zero_point_cpu.unsqueeze(-1) if zero_point_cpu is not None else None

        # CPU path
        cpu_out = _quantize_dequantize(
            x=x_3d.clone(),
            scale=scale_3d.clone(),
            zero_point=zp_3d.clone() if zp_3d is not None else None,
            q_min=q_min_cpu,
            q_max=q_max_cpu,
            args=args,
            global_scale=global_scale_cpu.clone() if global_scale_cpu is not None else None,
        )

        # CUDA path
        cuda_out = _quantize_dequantize(
            x=x_3d.cuda(),
            scale=scale_3d.cuda(),
            zero_point=zp_3d.cuda() if zp_3d is not None else None,
            q_min=q_min_cpu.cuda(),
            q_max=q_max_cpu.cuda(),
            args=args,
            global_scale=global_scale_cpu.cuda() if global_scale_cpu is not None else None,
        )
    else:
        # TENSOR/CHANNEL: shapes work directly
        # CPU path
        cpu_out = _quantize_dequantize(
            x=x_cpu.clone(),
            scale=scale_cpu.clone(),
            zero_point=zero_point_cpu.clone() if zero_point_cpu is not None else None,
            q_min=q_min_cpu,
            q_max=q_max_cpu,
            args=args,
            global_scale=global_scale_cpu.clone() if global_scale_cpu is not None else None,
        )

        # CUDA path
        cuda_out = _quantize_dequantize(
            x=x_cpu.cuda(),
            scale=scale_cpu.cuda(),
            zero_point=zero_point_cpu.cuda() if zero_point_cpu is not None else None,
            q_min=q_min_cpu.cuda(),
            q_max=q_max_cpu.cuda(),
            args=args,
            global_scale=global_scale_cpu.cuda() if global_scale_cpu is not None else None,
        )

    # Fixed tolerances per quantization type:
    #
    # Error comes from rounding differences (torch.round vs tl.rint) and FP precision.
    # Error in output = (rounding_diff) × scale, where rounding_diff ≤ 1 step.
    #
    # INT4/INT8: uniform spacing, max rounding diff = 0.5 steps
    #   - With test scales [0.01, 0.11], max error ≈ 0.11
    #   - atol = 0.15 covers this with buffer
    #
    # FP4: non-uniform spacing (gaps: 0.5 to 2.0), max rounding diff = 1.0 step
    #   - atol = 0.25 for larger potential errors
    #
    # FP8: falls back to CPU, should be identical
    #   - atol = 1e-5
    #
    # rtol = 0.01 handles any relative precision issues for larger values
    if type == "int":
        computed_atol = 0.15
        computed_rtol = 0.01
    elif type == "float" and num_bits == 4:
        computed_atol = 0.25
        computed_rtol = 0.01
    else:  # FP8 - CPU fallback
        computed_atol = 1e-5
        computed_rtol = 1e-5

    print("type: ", type)
    print("num_bits: ", num_bits)
    print("cpu out: ", cpu_out)
    print("cuda out: ", cuda_out.cpu())
    print("max diff: ", (cpu_out - cuda_out.cpu()).abs().max().item())
    print("*")

    assert torch.allclose(
        cpu_out, cuda_out.cpu(), rtol=computed_rtol, atol=computed_atol
    ), f"Mismatch: max diff = {(cpu_out - cuda_out.cpu()).abs().max().item()}, rtol = {computed_rtol}, atol = {computed_atol}"



@pytest.mark.parametrize(
    "num_bits,type,symmetric,global_scale,group_size",
    [
        # Tensor-level quantization (group_size=None)
        (8, "int", True, None, None),
        (8, "int", False, None, None),
        (4, "int", True, None, None),
        (4, "float", True, None, None),  # FP4
        (8, "float", True, None, None),
        (8, "float", True, torch.tensor([2.0]), None),
        (8, "int", False, torch.tensor([2.0]), None),
        # Group quantization
        (8, "int", True, None, 128),
        (8, "int", False, None, 128),
        (4, "int", True, None, 128),
        (4, "float", True, None, 128),  # FP4
        (8, "float", True, None, 128),
        (8, "float", True, torch.tensor([2.0]), 128),
        (8, "int", False, torch.tensor([2.0]), 128),
        (8, "int", True, None, 64),
        (8, "int", False, None, 256),
    ],
)
def test_quantize_triton_matches_cpu(
    num_bits, type, symmetric, global_scale, group_size
):
    """Verify that the Triton kernel (CUDA) produces identical output
    to the non-Triton (CPU) codepath for _quantize."""
    if not torch.accelerator.is_available():
        pytest.skip("CUDA not available")

    num_rows = 512
    num_cols = 1024

    if group_size is None:
        strategy = QuantizationStrategy.TENSOR
        args = QuantizationArgs(
            num_bits=num_bits,
            type=type,
            symmetric=symmetric,
            strategy=strategy,
        )
        x_cpu = torch.randn(num_rows, num_cols)
        scale_cpu = torch.rand(1) * 0.01 + 0.001
        zero_point_cpu = None if symmetric else torch.tensor([3.0])
    else:
        strategy = QuantizationStrategy.GROUP
        num_groups = num_cols // group_size
        args = QuantizationArgs(
            num_bits=num_bits,
            type=type,
            symmetric=symmetric,
            strategy=strategy,
            group_size=group_size,
        )
        x_cpu = torch.randn(num_rows, num_groups, group_size)
        scale_cpu = torch.rand(num_rows, num_groups, 1) * 0.01 + 0.001
        zero_point_cpu = (
            None if symmetric else torch.zeros(num_rows, num_groups, 1) + 3.0
        )

    global_scale_cpu = global_scale.clone() if global_scale is not None else None
    q_min_cpu, q_max_cpu = calculate_range(args, torch.device("cpu"))

    # Copy to CUDA and run Triton path
    x_cuda = x_cpu.cuda()
    scale_cuda = scale_cpu.cuda()
    zero_point_cuda = zero_point_cpu.cuda() if zero_point_cpu is not None else None
    global_scale_cuda = (
        global_scale_cpu.cuda() if global_scale_cpu is not None else None
    )
    q_min_cuda, q_max_cuda = calculate_range(args, torch.device("cuda"))

    num_rows = x_cuda.shape[0]
    scale_cuda, zero_point_cuda = adapt_scale_and_zp_for_triton(
        scale_cuda, zero_point_cuda, num_rows
    )

    # Run CPU (non-Triton) path
    cpu_out = _quantize(
        x=x_cpu,
        scale=scale_cpu,
        zero_point=zero_point_cpu,
        q_min=q_min_cpu,
        q_max=q_max_cpu,
        args=args,
        global_scale=global_scale_cpu,
        do_triton=False,
    )

    cuda_out = _quantize(
        x=x_cuda,
        scale=scale_cuda,
        zero_point=zero_point_cuda,
        q_min=q_min_cuda,
        q_max=q_max_cuda,
        args=args,
        global_scale=global_scale_cuda,
        do_triton=True,
    )

    # Compare results (bring CUDA output back to CPU)
    cuda_out_cpu = cuda_out.cpu()

    # For int types, there are edge cases where a <1e-5 difference gets
    # rounded to a different integer on CPU and GPU.
    if type == "int":
        atol, rtol = 1.0, 0
    else:
        atol, rtol = 1e-5, 0.15

    assert torch.allclose(cpu_out, cuda_out_cpu, atol=atol, rtol=rtol), (
        f"Mismatch between CPU and Triton paths: max diff = "
        f"{(cpu_out - cuda_out_cpu).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "num_bits,type,symmetric,global_scale,group_size",
    [
        # Tensor-level quantization (group_size=None)
        (8, "int", True, None, None),
        (8, "int", False, None, None),
        (4, "int", True, None, None),
        (4, "float", True, None, None),  # FP4
        (8, "float", True, None, None),
        (8, "float", True, torch.tensor([2.0]), None),
        (8, "int", False, torch.tensor([2.0]), None),
        # Group quantization
        (8, "int", True, None, 128),
        (8, "int", False, None, 128),
        (4, "int", True, None, 128),
        (4, "float", True, None, 128),  # FP4
        (8, "float", True, None, 128),
        (8, "float", True, torch.tensor([2.0]), 128),
        (8, "int", False, torch.tensor([2.0]), 128),
        (8, "int", True, None, 64),
        (8, "int", False, None, 256),
    ],
)
def test_quantize_triton_matches_cpu_non_contiguous(
    num_bits, type, symmetric, global_scale, group_size
):
    """Verify that the Triton kernel (CUDA) produces identical output
    to the non-Triton (CPU) codepath for _quantize with non-contiguous tensors."""
    if not torch.accelerator.is_available():
        pytest.skip("CUDA not available")

    num_rows = 512
    num_cols = 1024

    if group_size is None:
        strategy = QuantizationStrategy.TENSOR
        args = QuantizationArgs(
            num_bits=num_bits,
            type=type,
            symmetric=symmetric,
            strategy=strategy,
        )
        # Create non-contiguous tensor via transpose
        x_base = torch.randn(num_cols, num_rows)
        x_cpu = x_base.t()
        assert not x_cpu.is_contiguous(), "Test requires non-contiguous tensor"
        scale_cpu = torch.rand(1) * 0.01 + 0.001
        zero_point_cpu = None if symmetric else torch.tensor([3.0])
    else:
        strategy = QuantizationStrategy.GROUP
        num_groups = num_cols // group_size
        args = QuantizationArgs(
            num_bits=num_bits,
            type=type,
            symmetric=symmetric,
            strategy=strategy,
            group_size=group_size,
        )
        # Create non-contiguous tensor via transpose of first two dimensions
        x_base = torch.randn(num_groups, num_rows, group_size)
        x_cpu = x_base.permute(1, 0, 2)
        assert not x_cpu.is_contiguous(), "Test requires non-contiguous tensor"
        # Also make scale non-contiguous
        scale_base = torch.rand(num_groups, num_rows, 1) * 0.01 + 0.001
        scale_cpu = scale_base.permute(1, 0, 2)
        assert not scale_cpu.is_contiguous(), "Test requires non-contiguous scale"
        zero_point_cpu = (
            None
            if symmetric
            else (torch.zeros(num_groups, num_rows, 1) + 3.0).permute(1, 0, 2)
        )
        if zero_point_cpu is not None:
            assert (
                not zero_point_cpu.is_contiguous()
            ), "Test requires non-contiguous zero_point"

    global_scale_cpu = global_scale.clone() if global_scale is not None else None
    q_min_cpu, q_max_cpu = calculate_range(args, torch.device("cpu"))

    x_cuda = x_cpu.cuda()
    scale_cuda = scale_cpu.cuda()
    zero_point_cuda = zero_point_cpu.cuda() if zero_point_cpu is not None else None
    global_scale_cuda = (
        global_scale_cpu.cuda() if global_scale_cpu is not None else None
    )
    q_min_cuda, q_max_cuda = calculate_range(args, torch.device("cuda"))

    assert not x_cuda.is_contiguous(), "CUDA tensor should be non-contiguous"

    # Adapt scale/zp for Triton
    num_rows = x_cuda.shape[0]
    scale_cuda, zero_point_cuda = adapt_scale_and_zp_for_triton(
        scale_cuda, zero_point_cuda, num_rows
    )

    cpu_out = _quantize(
        x=x_cpu,
        scale=scale_cpu,
        zero_point=zero_point_cpu,
        q_min=q_min_cpu,
        q_max=q_max_cpu,
        args=args,
        global_scale=global_scale_cpu,
        do_triton=False,
    )

    cuda_out = _quantize(
        x=x_cuda,
        scale=scale_cuda,
        zero_point=zero_point_cuda,
        q_min=q_min_cuda,
        q_max=q_max_cuda,
        args=args,
        global_scale=global_scale_cuda,
        do_triton=True,
    )

    cuda_out_cpu = cuda_out.cpu()

    if type == "int":
        atol, rtol = 1.0, 0
    else:
        atol, rtol = 1e-5, 0.15

    assert torch.allclose(cpu_out, cuda_out_cpu, atol=atol, rtol=rtol), (
        f"Mismatch between CPU and Triton paths (non-contiguous): max diff = "
        f"{(cpu_out - cuda_out_cpu).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "num_block_rows,num_block_cols,block_structure",
    [
        (16, 16, [128, 128]),
        (2, 16, [128, 128]),
        (44, 16, [128, 128]),
        (16, 44, [128, 128]),
    ],
)
def test_quantize_triton_matches_cpu_block_4d(
    num_block_rows, num_block_cols, block_structure
):
    """Verify that the Triton kernel (CUDA) produces identical output
    to the non-Triton (CPU) codepath for _quantize with 4D block quantization.
    """
    if not torch.accelerator.is_available():
        pytest.skip("CUDA not available")

    num_bits = 8
    type_ = "float"
    symmetric = True

    args = QuantizationArgs(
        num_bits=num_bits,
        type=type_,
        symmetric=symmetric,
        strategy=QuantizationStrategy.BLOCK,
        block_structure=block_structure,
    )

    bh, bw = block_structure
    rows_2d = num_block_rows * bh
    cols_2d = num_block_cols * bw

    x_2d = torch.randn(rows_2d, cols_2d)
    x_cpu = x_2d.view(num_block_rows, bh, num_block_cols, bw).permute(0, 2, 1, 3)

    expected_stride = (bh * cols_2d, bw, cols_2d, 1)
    assert (
        x_cpu.stride() == expected_stride
    ), f"Stride mismatch: got {x_cpu.stride()}, expected {expected_stride}"
    assert not x_cpu.is_contiguous(), "Test requires non-contiguous tensor"

    scale_cpu = torch.rand(num_block_rows, num_block_cols, 1, 1) * 0.01 + 0.001

    q_min_cpu, q_max_cpu = calculate_range(args, torch.device("cpu"))

    x_cuda = x_cpu.cuda()
    scale_cuda = scale_cpu.cuda()
    q_min_cuda, q_max_cuda = calculate_range(args, torch.device("cuda"))

    assert not x_cuda.is_contiguous(), "CUDA tensor should be non-contiguous"
    assert x_cuda.stride() == expected_stride, "CUDA tensor stride should match"

    cpu_out = _quantize(
        x=x_cpu,
        scale=scale_cpu,
        zero_point=None,
        q_min=q_min_cpu,
        q_max=q_max_cpu,
        args=args,
        global_scale=None,
        do_triton=False,
    )

    cuda_out = _quantize(
        x=x_cuda,
        scale=scale_cuda,
        zero_point=None,
        q_min=q_min_cuda,
        q_max=q_max_cuda,
        args=args,
        global_scale=None,
        do_triton=True,
    )

    cuda_out_cpu = cuda_out.cpu()

    # FP8 tolerance
    atol, rtol = 1e-5, 0.15

    assert torch.allclose(
        cpu_out.float(), cuda_out_cpu.float(), atol=atol, rtol=rtol
    ), (
        f"Mismatch between CPU and Triton paths (4D block): max diff = "
        f"{(cpu_out.float() - cuda_out_cpu.float()).abs().max().item()}"
    )
