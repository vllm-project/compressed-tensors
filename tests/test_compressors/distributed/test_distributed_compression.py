# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for distributed model compression."""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from compressed_tensors.compressors.model_compressors import ModelCompressor
from compressed_tensors.offload import offload_module
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


class TwoLayerModel(nn.Module):
    """Simple model for testing distributed compression."""

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
@torchrun(world_size=2)
def test_distributed_model_compression():
    """Test end-to-end distributed model compression."""
    model = TwoLayerModel()
    setup_quantized_model(model)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress the model
    compressor.compress_model(model)

    # Verify compression happened
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model.layer2, "weight_packed")
    assert model.layer1.weight_packed.dtype == torch.int32
    assert model.layer2.weight_packed.dtype == torch.int32

    # Verify compression status is updated
    assert (
        compressor.quantization_config.quantization_status
        == QuantizationStatus.COMPRESSED
    )


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_consistency():
    """Test that all ranks have consistent state after distributed compression."""
    model = TwoLayerModel()
    setup_quantized_model(model)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress the model
    compressor.compress_model(model)
    dist.barrier()

    # Compute checksums for each layer
    layer1_checksum = model.layer1.weight_packed.sum().item()
    layer2_checksum = model.layer2.weight_packed.sum().item()

    # Gather checksums from all ranks
    if dist.get_rank() == 0:
        gathered_layer1 = [None] * dist.get_world_size()
        gathered_layer2 = [None] * dist.get_world_size()
        dist.gather_object(layer1_checksum, gathered_layer1, dst=0)
        dist.gather_object(layer2_checksum, gathered_layer2, dst=0)

        # All ranks should have identical checksums
        for i in range(1, dist.get_world_size()):
            assert gathered_layer1[i] == gathered_layer1[0], (
                f"Layer1 mismatch: rank {i} has {gathered_layer1[i]}, "
                f"rank 0 has {gathered_layer1[0]}"
            )
            assert gathered_layer2[i] == gathered_layer2[0], (
                f"Layer2 mismatch: rank {i} has {gathered_layer2[i]}, "
                f"rank 0 has {gathered_layer2[0]}"
            )
    else:
        dist.gather_object(layer1_checksum, None, dst=0)
        dist.gather_object(layer2_checksum, None, dst=0)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_with_offload():
    """Test distributed compression with offloaded modules."""
    model = TwoLayerModel()
    setup_quantized_model(model)

    # Offload model to CPU
    offload_module(model.layer1, onload_device="cuda", offload_device="cpu")
    offload_module(model.layer2, onload_device="cuda", offload_device="cpu")

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress the model
    compressor.compress_model(model)

    # Verify compression happened even with offloading
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model.layer2, "weight_packed")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_decompress_roundtrip():
    """Test that distributed compression + decompression preserves values."""
    model = TwoLayerModel()
    setup_quantized_model(model)

    # Store original weights
    original_layer1 = model.layer1.weight.data.clone()
    original_layer2 = model.layer2.weight.data.clone()

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress and decompress
    compressor.compress_model(model)
    dist.barrier()
    compressor.decompress_model(model)
    dist.barrier()

    # Weights should be back to float
    assert model.layer1.weight.dtype == torch.float32
    assert model.layer2.weight.dtype == torch.float32

    # Values should be close (within quantization error)
    diff1 = torch.abs(original_layer1 - model.layer1.weight.data)
    diff2 = torch.abs(original_layer2 - model.layer2.weight.data)
    assert torch.max(diff1) < 1.0
    assert torch.max(diff2) < 1.0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_many_layers():
    """Test distributed compression with many layers to ensure load balancing."""

    class ManyLayerModel(nn.Module):
        def __init__(self, num_layers=10):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(10, 10, bias=False) for _ in range(num_layers)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = ManyLayerModel(num_layers=10)
    setup_quantized_model(model)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress the model
    compressor.compress_model(model)
    dist.barrier()

    # All layers should be compressed
    for layer in model.layers:
        assert hasattr(layer, "weight_packed")
        assert layer.weight_packed.dtype == torch.int32


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_skips_non_quantized():
    """Test that non-quantized layers are skipped in distributed compression."""

    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.quantized = nn.Linear(10, 10, bias=False)
            self.non_quantized = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            x = self.quantized(x)
            x = self.non_quantized(x)
            return x

    model = MixedModel()

    # Only quantize one layer
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            strategy="channel",
            symmetric=True,
            type="int",
        ),
    )
    model.quantized.quantization_scheme = scheme
    model.quantized.quantization_status = QuantizationStatus.FROZEN
    model.quantized.weight_scale = nn.Parameter(
        torch.ones(model.quantized.weight.shape[0], 1) * 0.01
    )
    model.quantized.weight_zero_point = nn.Parameter(
        torch.zeros(model.quantized.weight.shape[0], 1, dtype=torch.int32),
        requires_grad=False,
    )

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Store original non-quantized weight dtype
    original_dtype = model.non_quantized.weight.dtype

    # Compress the model
    compressor.compress_model(model)

    # Quantized layer should be compressed
    assert hasattr(model.quantized, "weight_packed")

    # Non-quantized layer should remain unchanged
    assert model.non_quantized.weight.dtype == original_dtype
    assert not hasattr(model.non_quantized, "weight_packed")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_empty_model():
    """Test distributed compression with an empty model."""
    model = nn.Sequential()

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Should not raise an error
    compressor.compress_model(model)
    dist.barrier()


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_single_layer():
    """Test distributed compression with a single layer."""

    class SingleLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 10, bias=False)

    model = SingleLayerModel()
    setup_quantized_model(model)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress the model
    compressor.compress_model(model)
    dist.barrier()

    # Layer should be compressed
    assert hasattr(model.layer, "weight_packed")

    # All ranks should have the same state
    checksum = model.layer.weight_packed.sum().item()
    if dist.get_rank() == 0:
        gathered = [None] * dist.get_world_size()
        dist.gather_object(checksum, gathered, dst=0)
        for i in range(1, dist.get_world_size()):
            assert gathered[i] == gathered[0]
    else:
        dist.gather_object(checksum, None, dst=0)
