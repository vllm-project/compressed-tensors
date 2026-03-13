# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import OrderedDict

import pytest
import torch
from compressed_tensors import FloatQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from torch.nn.modules import Linear, Sequential


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.tensor([index // group_size for index in range(columns)])[perm]


@pytest.mark.parametrize(
    "strategy,group_size,sc,zp",
    [
        [QuantizationStrategy.TENSOR, None, torch.tensor(0.01), torch.tensor(0)],
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
    module_sd = {
        "weight": torch.rand((512, 1024)),
        "weight_scale": sc.to(torch.float32),
        "weight_zero_point": zp.to(torch.float32),
    }
    if group_size is not None:
        module_sd["weight_g_idx"] = make_dummy_g_idx(1024, group_size)

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            strategy=strategy, type="float", group_size=group_size
        ),
    )

    compressed = FloatQuantizationCompressor.compress(module_sd, scheme=scheme)

    # symmetric → zero_point is dropped
    assert "weight_zero_point" not in compressed
    assert compressed["weight_scale"].dtype == torch.float32
    assert torch.equal(compressed["weight_scale"], module_sd["weight_scale"])
    if group_size is not None:
        assert torch.equal(compressed["weight_g_idx"], module_sd["weight_g_idx"])


@pytest.mark.parametrize(
    "strategy,group_size",
    [
        [QuantizationStrategy.TENSOR, None],
        [QuantizationStrategy.CHANNEL, None],
    ],
)
def test_compress_decompress_match(
    mock_per_group_calibration,
    mock_per_channel_calibration,
    strategy,
    group_size,
):
    model = Sequential(OrderedDict([("dummy", Linear(512, 1024, bias=None))]))
    quant_config = QuantizationConfig(
        config_groups={
            "group_1": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    strategy=strategy, type="float", group_size=group_size
                ),
            )
        },
        ignore=["lm_head"],
    )
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

    scheme = quant_config.config_groups["group_1"]

    # Build per-module state dict
    module_sd = {
        name: param.data.clone() for name, param in model.dummy.named_parameters()
    }

    compressed = FloatQuantizationCompressor.compress(module_sd, scheme=scheme)
    decompressed = FloatQuantizationCompressor.decompress(compressed, scheme=scheme)

    fake_quant_dummy = fake_quantize(
        model.dummy.weight,
        scale=model.dummy.weight_scale,
        zero_point=model.dummy.weight_zero_point,
        args=scheme.weights,
    )
    assert torch.equal(fake_quant_dummy, decompressed["weight"])


@pytest.mark.parametrize(
    "rows,cols,block_height,block_width",
    [
        (10944, 2048, 128, 128),
        (2048, 10944, 128, 128),
        (256, 256, 128, 128),
        (300, 400, 128, 128),
        (256, 300, 128, 128),
        (300, 256, 128, 128),
    ],
)
def test_block_quant_compression_padding(rows, cols, block_height, block_width):
    """
    Block quantization compresses weights with non-divisible dimensions without
    changing the original shape.
    """
    block_structure = [block_height, block_width]
    num_rb = math.ceil(rows / block_height)
    num_cb = math.ceil(cols / block_width)

    module_sd = {
        "weight": torch.rand((rows, cols)),
        "weight_scale": torch.rand((num_rb, num_cb)) * 0.01 + 0.001,
        "weight_zero_point": torch.zeros((num_rb, num_cb)),
    }

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            strategy=QuantizationStrategy.BLOCK,
            type="float",
            block_structure=block_structure,
        ),
    )

    compressed = FloatQuantizationCompressor.compress(module_sd, scheme=scheme)

    # Compressed weight should retain the original shape
    assert compressed["weight"].shape == (
        rows,
        cols,
    ), "Compressed weight shape should match original"
