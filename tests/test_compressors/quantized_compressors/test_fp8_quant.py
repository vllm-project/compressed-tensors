# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
from collections import OrderedDict

import pytest
import torch
from compressed_tensors import FloatQuantizationCompressor
from compressed_tensors.compressors.quantized_compressors.naive_quantized import (
    MXFP8QuantizationCompressor,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams
from safetensors.torch import save_file
from torch.nn.modules import Linear, Sequential


def get_dummy_quant_config(strategy, group_size=None):
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                strategy=strategy, type="float", group_size=group_size
            ),
        ),
    }
    ignore = ["lm_head"]
    quant_config = QuantizationConfig(
        config_groups=config_groups,
        ignore=ignore,
    )

    return quant_config


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.tensor([index // group_size for index in range(columns)])[perm]


@pytest.mark.parametrize(
    "strategy,group_size,sc,zp",
    [
        [QuantizationStrategy.TENSOR, None, 0.01, 0],
        [
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8)) * 0.01,
            torch.zeros((512, 8), dtype=torch.int8),
        ],
        [
            QuantizationStrategy.CHANNEL,
            None,
            torch.rand((512, 1)) * 0.01,
            torch.zeros((512, 1), dtype=torch.int8),
        ],
    ],
)
def test_quant_format(strategy, group_size, sc, zp):
    dense_state_dict = {
        "dummy.weight": torch.rand((512, 1024)),
        "dummy.weight_scale": torch.tensor(sc, dtype=torch.float32),
        "dummy.weight_zero_point": torch.tensor(zp, dtype=torch.float32),
    }
    if group_size is not None:
        dense_state_dict["dummy.weight_g_idx"] = make_dummy_g_idx(1024, group_size)

    quant_config = get_dummy_quant_config(strategy=strategy, group_size=group_size)

    compressor = FloatQuantizationCompressor(config=quant_config)
    module_name_to_scheme = {"dummy": quant_config.config_groups["group_1"]}
    compressed_state_dict = compressor.compress(
        dense_state_dict, names_to_scheme=module_name_to_scheme
    )

    # state_dict params should be the same, minus the zero_point if symmetric
    assert len(dense_state_dict) == len(compressed_state_dict) + 1

    # check compressed to int8
    assert compressed_state_dict["dummy.weight_scale"].dtype == torch.float32
    assert torch.equal(
        compressed_state_dict["dummy.weight_scale"],
        dense_state_dict["dummy.weight_scale"],
    )
    if group_size is not None:
        assert torch.equal(
            compressed_state_dict["dummy.weight_g_idx"],
            dense_state_dict["dummy.weight_g_idx"],
        )


@pytest.mark.parametrize(
    "strategy,group_size",
    [
        [QuantizationStrategy.TENSOR, None],
        [QuantizationStrategy.CHANNEL, None],
        # Note that group quantization is not supported
    ],
)
def test_reload_match(
    mock_per_group_calibration,
    mock_per_channel_calibration,
    strategy,
    group_size,
    tmp_path,
):
    model = Sequential(
        OrderedDict(
            [
                ("dummy", Linear(512, 1024, bias=None)),
            ]
        )
    )
    quant_config = get_dummy_quant_config(strategy=strategy, group_size=group_size)
    apply_quantization_config(model, quant_config)
    model.dummy.quantization_status = QuantizationStatus.CALIBRATION
    if strategy == QuantizationStrategy.GROUP:
        mock_per_group_calibration(
            model.dummy, base_name="weight", value=model.dummy.weight, group_size=128
        )
    if strategy == QuantizationStrategy.CHANNEL:
        mock_per_channel_calibration(
            model.dummy, base_name="weight", value=model.dummy.weight
        )

    compressor = FloatQuantizationCompressor(config=quant_config)
    module_name_to_scheme = {
        "dummy": quant_config.config_groups["group_1"],
    }
    compressed_state_dict = compressor.compress(
        model.state_dict(), names_to_scheme=module_name_to_scheme
    )
    save_file(compressed_state_dict, tmp_path / "model.safetensors")
    reconstructed_dense_gen = compressor.decompress(
        tmp_path, names_to_scheme=module_name_to_scheme
    )
    reconstructed_dense = {}
    for name, value in reconstructed_dense_gen:
        reconstructed_dense[name] = value

    fake_quant_dummy = fake_quantize(
        model.dummy.weight,
        scale=model.dummy.weight_scale,
        zero_point=model.dummy.weight_zero_point,
        args=module_name_to_scheme["dummy"].weights,
    )
    assert torch.equal(fake_quant_dummy, reconstructed_dense["dummy"].get("weight"))

    shutil.rmtree(tmp_path)


def get_block_quant_config(block_structure):
    """Create a quantization config for block quantization."""
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                strategy=QuantizationStrategy.BLOCK,
                type="float",
                block_structure=block_structure,
            ),
        ),
    }
    return QuantizationConfig(config_groups=config_groups)


@pytest.mark.parametrize(
    "rows,cols,block_height,block_width",
    [
        (10944, 2048, 128, 128),  # DeepSeek-V2-Lite intermediate_size
        (2048, 10944, 128, 128),  # DeepSeek-V2-Lite down_proj
        (256, 256, 128, 128),  # Divisible dimensions (should not pad)
        (300, 400, 128, 128),  # Both non-divisible
        (256, 300, 128, 128),  # Only cols non-divisible
        (300, 256, 128, 128),  # Only rows non-divisible
    ],
)
def test_block_quant_compression_padding(rows, cols, block_height, block_width):
    """
    Test that block quantization compresses weights with non-divisible dimensions
    without changing the original shape
    """
    import math

    block_structure = [block_height, block_width]

    # Create scale tensor with ceiling division
    num_rb = math.ceil(rows / block_height)
    num_cb = math.ceil(cols / block_width)

    dense_state_dict = {
        "dummy.weight": torch.rand((rows, cols)),
        "dummy.weight_scale": torch.rand((num_rb, num_cb)) * 0.01 + 0.001,
        "dummy.weight_zero_point": torch.zeros((num_rb, num_cb)),
    }

    quant_config = get_block_quant_config(block_structure)
    compressor = FloatQuantizationCompressor(config=quant_config)
    module_name_to_scheme = {"dummy": quant_config.config_groups["group_1"]}

    compressed_state_dict = compressor.compress(
        dense_state_dict, names_to_scheme=module_name_to_scheme
    )

    # Check that weight was padded if needed
    compressed_weight = compressed_state_dict["dummy.weight"]

    # Compressed weight should retain shape of original
    assert compressed_weight.shape == (
        rows,
        cols,
    ), "Compressed weight shape should be the same as original weight shape"


def test_mxfp8_compress_decompress():
    """
    Test MXFP8 compress/decompress round-trip with group strategy
    and group_size=32. Verifies weights survive the cycle (lossy but close).
    """
    rows, cols = 512, 1024
    group_size = 32
    num_groups = cols // group_size

    quant_args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
        symmetric=True,
    )

    weight = torch.randn((rows, cols))

    # Compute scales using calculate_qparams (which generates MX scales)
    reshaped = weight.reshape(rows, num_groups, group_size)
    min_vals = reshaped.amin(dim=-1)
    max_vals = reshaped.amax(dim=-1)
    scale, zp = calculate_qparams(min_vals, max_vals, quant_args)

    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=quant_args,
        ),
    }
    quant_config = QuantizationConfig(config_groups=config_groups)
    compressor = MXFP8QuantizationCompressor(config=quant_config)

    # Build state dict
    dense_state_dict = {
        "dummy.weight": weight,
        "dummy.weight_scale": scale,
        "dummy.weight_zero_point": zp,
    }

    module_name_to_scheme = {"dummy": quant_config.config_groups["group_1"]}

    # Compress
    compressed_state_dict = compressor.compress(
        dense_state_dict, names_to_scheme=module_name_to_scheme
    )

    # Check compressed weight is FP8
    compressed_weight = compressed_state_dict["dummy.weight"]
    assert compressed_weight.dtype == torch.float8_e4m3fn

    # Check scale is stored as uint8 (E8M0 exponent format)
    compressed_scale = compressed_state_dict["dummy.weight_scale"]
    assert compressed_scale.dtype == torch.uint8

    # Decompress
    decompressed_gen = compressor.decompress_from_state_dict(
        compressed_state_dict, names_to_scheme=module_name_to_scheme
    )
    for name, data in decompressed_gen:
        decompressed_weight = data["weight"]

    # Check shapes match
    assert decompressed_weight.shape == weight.shape

    # FP8 quantization is lossy, but should be reasonably close
    # Use a tolerance that accounts for FP8 precision limits
    assert torch.allclose(decompressed_weight.float(), weight, atol=0.1, rtol=0.1)
