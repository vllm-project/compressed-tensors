# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for distributed model decompression."""

import pytest
import torch
import torch.nn as nn
from compressed_tensors.compressors.model_compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


class TwoLayerModel(nn.Module):
    """Simple model for testing distributed decompression."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10, bias=False)
        self.layer2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def create_quantization_config(bits=4, format="pack-quantized"):
    """Helper to create a QuantizationConfig for testing."""
    config_dict = {
        "format": format,
        "global_compression_ratio": 1.0,
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": bits,
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
    }
    return QuantizationConfig.model_validate(config_dict)


def setup_quantized_model(model: nn.Module, bits: int = 4) -> nn.Module:
    """Set up a model with quantization schemes and parameters."""
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits,
            strategy="channel",
            symmetric=True,
            type="int",
        ),
    )

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.quantization_scheme = scheme
            module.quantization_status = QuantizationStatus.FROZEN
            module.weight_scale = nn.Parameter(
                torch.ones(module.weight.shape[0], 1) * 0.01
            )
            module.weight_zero_point = nn.Parameter(
                torch.zeros(module.weight.shape[0], 1, dtype=torch.int32),
                requires_grad=False,
            )

    return model


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_distributed_model_decompression():
    """Test end-to-end distributed model decompression."""
    model = TwoLayerModel()
    setup_quantized_model(model)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress the model
    compressor.compress_model(model)

    # Decompress model
    compressor.decompress_model(model)

    # Verify decompression happened
    assert hasattr(model.layer1, "weight")
    assert hasattr(model.layer2, "weight")
    assert model.layer1.weight.dtype == torch.float32
    assert model.layer2.weight.dtype == torch.float32

    # Verify compression status is updated
    assert (
        compressor.quantization_config.quantization_status
        == QuantizationStatus.DECOMPRESSED
    )
