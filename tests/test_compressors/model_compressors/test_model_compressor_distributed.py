# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for distributed compression in ModelCompressor."""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from compressed_tensors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


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


def setup_quantized_module(module: nn.Linear, bits: int = 4):
    """Set up a linear module with quantization scheme and parameters."""
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits,
            strategy="channel",
            symmetric=True,
            type="int",
        ),
    )

    module.quantization_scheme = scheme
    module.quantization_status = QuantizationStatus.FROZEN
    module.weight_scale = nn.Parameter(torch.ones(module.weight.shape[0], 1) * 0.01)
    module.weight_zero_point = nn.Parameter(
        torch.zeros(module.weight.shape[0], 1, dtype=torch.int32),
        requires_grad=False,
    )


class TwoLayerModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10, bias=False)
        self.layer2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_model_distributed_basic():
    """Test basic distributed compression via ModelCompressor."""
    model = TwoLayerModel()
    setup_quantized_module(model.layer1)
    setup_quantized_module(model.layer2)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress the model
    compressor.compress_model(model)

    # Verify compression happened
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model.layer2, "weight_packed")
    assert model.layer1.weight_packed.dtype == torch.int32
    assert model.layer2.weight_packed.dtype == torch.int32

    # Verify status updated
    assert compressor.quantization_config.quantization_status == (
        QuantizationStatus.COMPRESSED
    )

    # Verify hook was added
    assert hasattr(model, "ct_decompress_hook")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_model_distributed_consistency():
    """Test that all ranks have consistent state after distributed compression."""
    model = TwoLayerModel()
    setup_quantized_module(model.layer1)
    setup_quantized_module(model.layer2)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Compress the model
    compressor.compress_model(model)
    dist.barrier()

    # Compute checksums
    layer1_sum = model.layer1.weight_packed.sum().item()
    layer2_sum = model.layer2.weight_packed.sum().item()

    # Gather checksums from all ranks
    if dist.get_rank() == 0:
        gathered_layer1 = [None] * dist.get_world_size()
        gathered_layer2 = [None] * dist.get_world_size()
        dist.gather_object(layer1_sum, gathered_layer1, dst=0)
        dist.gather_object(layer2_sum, gathered_layer2, dst=0)

        # All ranks should have identical checksums
        for i in range(1, dist.get_world_size()):
            assert (
                gathered_layer1[i] == gathered_layer1[0]
            ), f"Layer1 mismatch between rank {i} and rank 0"
            assert (
                gathered_layer2[i] == gathered_layer2[0]
            ), f"Layer2 mismatch between rank {i} and rank 0"
    else:
        dist.gather_object(layer1_sum, None, dst=0)
        dist.gather_object(layer2_sum, None, dst=0)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_model_distributed_no_quantized_modules():
    """Test distributed compression with no quantized modules."""
    model = TwoLayerModel()
    # Don't set up quantization schemes

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    # Should not raise
    compressor.compress_model(model)

    # Should not have compressed weights
    assert not hasattr(model.layer1, "weight_packed")
    assert not hasattr(model.layer2, "weight_packed")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_model_distributed_partial_quantization():
    """Test distributed compression with only some modules quantized."""
    model = TwoLayerModel()
    setup_quantized_module(model.layer1)
    # Don't quantize layer2

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    compressor.compress_model(model)

    # Only layer1 should be compressed
    assert hasattr(model.layer1, "weight_packed")
    assert not hasattr(model.layer2, "weight_packed")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_decompress_distributed_roundtrip():
    """Test compress then decompress in distributed mode."""
    model = TwoLayerModel()
    setup_quantized_module(model.layer1)
    setup_quantized_module(model.layer2)

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

    # Hook should be removed
    assert not hasattr(model, "ct_decompress_hook")

    # Status should be updated
    assert compressor.quantization_config.quantization_status == (
        QuantizationStatus.DECOMPRESSED
    )

    # Values should be close (within quantization error)
    diff1 = torch.abs(original_layer1 - model.layer1.weight.data)
    diff2 = torch.abs(original_layer2 - model.layer2.weight.data)
    assert torch.max(diff1) < 1.0
    assert torch.max(diff2) < 1.0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_model_distributed_many_layers():
    """Test distributed compression with many layers for load balancing."""

    class ManyLayerModel(nn.Module):
        def __init__(self, num_layers=10):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(10, 10, bias=False) for _ in range(num_layers)]
            )

    model = ManyLayerModel(num_layers=10)
    for layer in model.layers:
        setup_quantized_module(layer)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    compressor.compress_model(model)
    dist.barrier()

    # All layers should be compressed
    for layer in model.layers:
        assert hasattr(layer, "weight_packed")
        assert layer.weight_packed.dtype == torch.int32

    # Check consistency across ranks
    checksums = [layer.weight_packed.sum().item() for layer in model.layers]

    if dist.get_rank() == 0:
        gathered = [None] * dist.get_world_size()
        dist.gather_object(checksums, gathered, dst=0)

        # All ranks should have identical checksums
        for i in range(1, dist.get_world_size()):
            for layer_idx, (c1, c2) in enumerate(zip(gathered[0], gathered[i])):
                assert c1 == c2, f"Layer {layer_idx} mismatch between ranks 0 and {i}"
    else:
        dist.gather_object(checksums, None, dst=0)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_model_distributed_force_format():
    """Test that force_compression_format works in distributed mode."""
    model = TwoLayerModel()
    setup_quantized_module(model.layer1)
    setup_quantized_module(model.layer2)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    # Force a specific format
    compressor = ModelCompressor(
        quantization_config=q_config, force_compression_format="pack-quantized"
    )

    compressor.compress_model(model)

    # Verify compression with forced format
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model.layer2, "weight_packed")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_model_distributed_from_pretrained():
    """Test from_pretrained_model entrypoint works with distributed compression."""
    model = TwoLayerModel()
    setup_quantized_module(model.layer1)
    setup_quantized_module(model.layer2)

    # Use from_pretrained_model
    compressor = ModelCompressor.from_pretrained_model(model)

    # Should infer format
    assert compressor.quantization_config is not None

    # Compress in distributed mode
    compressor.compress_model(model)
    dist.barrier()

    # Verify compression
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model.layer2, "weight_packed")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_compress_model_distributed_hook_triggers():
    """Test that decompression hook triggers correctly in distributed mode."""
    model = TwoLayerModel()
    setup_quantized_module(model.layer1)
    setup_quantized_module(model.layer2)

    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)

    compressor.compress_model(model)
    dist.barrier()

    # Verify compressed
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model, "ct_decompress_hook")

    # Forward pass should trigger decompression
    x = torch.randn(2, 10)
    _ = model(x)

    # After forward, should be decompressed
    assert model.layer1.weight.dtype == torch.float32
    assert not hasattr(model.layer1, "weight_packed")
    assert not hasattr(model, "ct_decompress_hook")
