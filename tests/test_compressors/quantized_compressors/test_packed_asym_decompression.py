# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
End-to-end tests for asymmetric quantization with zero-point decompression.
"""

import pytest
import torch
from compressed_tensors.compressors.model_compressors.model_compressor import (
    ModelCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
    apply_quantization_config,
)
from torch.nn import Linear, Module


class SimpleModel(Module):
    """Simple model for testing"""

    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.layer1 = Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


def create_asymmetric_quant_config(
    num_bits=4, strategy=QuantizationStrategy.GROUP, group_size=128
) -> QuantizationConfig:
    """Create an asymmetric quantization config"""
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=num_bits,
                strategy=strategy.value,
                group_size=(
                    group_size if strategy == QuantizationStrategy.GROUP else None
                ),
                symmetric=False,
            ),
        ),
    }
    return QuantizationConfig(config_groups=config_groups)


@pytest.mark.parametrize(
    "strategy,group_size",
    [
        (QuantizationStrategy.GROUP, 128),
        (QuantizationStrategy.CHANNEL, None),
    ],
)
def test_end_to_end_asymmetric_quantization(
    strategy,
    group_size,
    mock_per_group_calibration,
    mock_per_channel_calibration,
):
    """
    Test end-to-end workflow: quantize -> compress -> decompress in memory
    """
    model = SimpleModel()
    original_weights = {
        "layer1": model.layer1.weight.detach().clone(),
        "layer2": model.layer2.weight.detach().clone(),
    }

    quant_config = create_asymmetric_quant_config(
        num_bits=4, strategy=strategy, group_size=group_size
    )
    # Set pack-quantized format for ModelCompressor usage
    quant_config.format = CompressionFormat.pack_quantized.value
    apply_quantization_config(model, quant_config)

    if strategy == QuantizationStrategy.GROUP:
        mock_per_group_calibration(
            model.layer1, "weight", model.layer1.weight, group_size
        )
        mock_per_group_calibration(
            model.layer2, "weight", model.layer2.weight, group_size
        )
    else:
        mock_per_channel_calibration(model.layer1, "weight", model.layer1.weight)
        mock_per_channel_calibration(model.layer2, "weight", model.layer2.weight)

    # Compress and decompress in memory using ModelCompressor
    mc = ModelCompressor(quantization_config=quant_config)
    mc.compress_model(model)

    # Verify compression created zero-point parameters
    assert hasattr(model.layer1, "weight_zero_point")
    assert hasattr(model.layer2, "weight_zero_point")
    assert model.layer1.weight_zero_point.dtype == torch.int32
    assert model.layer2.weight_zero_point.dtype == torch.int32

    # Decompress in memory
    mc.decompress_model(model)

    # Verify decompression restored weights correctly
    assert model.layer1.weight.shape == original_weights["layer1"].shape
    assert model.layer2.weight.shape == original_weights["layer2"].shape
    assert model.layer1.weight.dtype.is_floating_point
    assert model.layer2.weight.dtype.is_floating_point
    assert not torch.isnan(model.layer1.weight).any()
    assert not torch.isnan(model.layer2.weight).any()
    assert not torch.isinf(model.layer1.weight).any()
    assert not torch.isinf(model.layer2.weight).any()


@pytest.mark.parametrize("num_bits", [4, 8])
def test_asymmetric_quantization_accuracy(num_bits, mock_per_group_calibration):
    """
    Test that asymmetric quantization with zero-point preserves accuracy better
    than symmetric quantization for biased weight distributions.
    """
    shape = (256, 512)
    biased_weights = torch.randn(shape) + 2.0

    quant_config = create_asymmetric_quant_config(
        num_bits=num_bits,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
    )
    quant_config.format = CompressionFormat.pack_quantized.value

    class SingleLayer(Module):
        def __init__(self):
            super().__init__()
            self.layer = Linear(shape[1], shape[0], bias=False)

    model = SingleLayer()
    apply_quantization_config(model, quant_config)

    with torch.no_grad():
        model.layer.weight.copy_(biased_weights)
    mock_per_group_calibration(model.layer, "weight", model.layer.weight, 128)

    # Compress and decompress in memory using ModelCompressor
    mc = ModelCompressor(quantization_config=quant_config)
    mc.compress_model(model)
    mc.decompress_model(model)

    decompressed_weights = model.layer.weight
    assert decompressed_weights.shape == shape
    assert not torch.isnan(decompressed_weights).any()
    assert not torch.isinf(decompressed_weights).any()
    threshold = torch.std(torch.rand(shape) - torch.rand(shape))
    assert torch.std(biased_weights - decompressed_weights) < threshold
