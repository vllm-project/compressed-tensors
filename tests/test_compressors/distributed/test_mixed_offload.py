# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for distributed compression with mixed offload types (device, cpu, disk)."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from compressed_tensors import ModelCompressor
from compressed_tensors.offload import dispatch_with_map
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


class ThreeLayerModel(nn.Module):
    """Three-layer model for testing mixed offload types."""

    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(10, 10, bias=False)
        self.layer1 = nn.Linear(10, 10, bias=False)
        self.layer2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def create_quantization_config(bits=4, format="pack-quantized"):
    """Helper to create a W4A16 QuantizationConfig."""
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
                # W4A16: only quantize weights, not activations
            }
        },
    }
    return QuantizationConfig.model_validate(config_dict)


def setup_quantized_module(module: nn.Linear, bits: int = 4):
    """Set up a linear module with W4A16 quantization scheme."""
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits,
            strategy="channel",
            symmetric=True,
            type="int",
        ),
        # No input_activations for W4A16
    )

    module.quantization_scheme = scheme
    module.quantization_status = QuantizationStatus.FROZEN
    module.weight_scale = nn.Parameter(torch.ones(module.weight.shape[0], 1) * 0.01)
    module.weight_zero_point = nn.Parameter(
        torch.zeros(module.weight.shape[0], 1, dtype=torch.int32),
        requires_grad=False,
    )


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_mixed_offload(tmp_path):
    """Test distributed compression with mixed offload types (device, cpu, disk)."""
    # Create offload directory
    offload_dir = tmp_path / "offload_dir"
    os.makedirs(offload_dir, exist_ok=True)

    # Create 3-layer model with W4A16 quantization
    model = ThreeLayerModel()
    setup_quantized_module(model.layer0, bits=4)
    setup_quantized_module(model.layer1, bits=4)
    setup_quantized_module(model.layer2, bits=4)

    # Setup mixed offload: layer0=device, layer1=cpu, layer2=disk
    device_map = {
        "layer0": (torch.device("cuda"), torch.device("cuda")),  # device
        "layer1": (torch.device("cuda"), torch.device("cpu")),  # cpu
        "layer2": (torch.device("cuda"), "disk"),  # disk
    }
    dispatch_with_map(model, device_map, str(offload_dir))

    # Compress the model
    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)
    compressor.compress_model(model)
    dist.barrier()

    # Verify all layers are compressed
    assert hasattr(model.layer0, "weight_packed")
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model.layer2, "weight_packed")
    assert model.layer0.weight_packed.dtype == torch.int32
    assert model.layer1.weight_packed.dtype == torch.int32
    assert model.layer2.weight_packed.dtype == torch.int32

    # Verify consistency across ranks
    layer0_sum = model.layer0.weight_packed.sum().item()
    layer1_sum = model.layer1.weight_packed.sum().item()
    layer2_sum = model.layer2.weight_packed.sum().item()

    if dist.get_rank() == 0:
        gathered_layer0 = [None] * dist.get_world_size()
        gathered_layer1 = [None] * dist.get_world_size()
        gathered_layer2 = [None] * dist.get_world_size()
        dist.gather_object(layer0_sum, gathered_layer0, dst=0)
        dist.gather_object(layer1_sum, gathered_layer1, dst=0)
        dist.gather_object(layer2_sum, gathered_layer2, dst=0)

        # All ranks should have identical checksums
        for i in range(1, dist.get_world_size()):
            assert (
                gathered_layer0[i] == gathered_layer0[0]
            ), f"Layer0 (device) mismatch between rank {i} and rank 0"
            assert (
                gathered_layer1[i] == gathered_layer1[0]
            ), f"Layer1 (cpu) mismatch between rank {i} and rank 0"
            assert (
                gathered_layer2[i] == gathered_layer2[0]
            ), f"Layer2 (disk) mismatch between rank {i} and rank 0"
    else:
        dist.gather_object(layer0_sum, None, dst=0)
        dist.gather_object(layer1_sum, None, dst=0)
        dist.gather_object(layer2_sum, None, dst=0)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_mixed_offload_roundtrip(tmp_path):
    """Test compress/decompress roundtrip with mixed offload types."""
    # Create offload directory
    offload_dir = tmp_path / "offload_dir"
    os.makedirs(offload_dir, exist_ok=True)

    # Create 3-layer model with W4A16 quantization
    model = ThreeLayerModel()
    setup_quantized_module(model.layer0, bits=4)
    setup_quantized_module(model.layer1, bits=4)
    setup_quantized_module(model.layer2, bits=4)

    # Store original weights
    original_layer0 = model.layer0.weight.data.clone()
    original_layer1 = model.layer1.weight.data.clone()
    original_layer2 = model.layer2.weight.data.clone()

    # Setup mixed offload: layer0=device, layer1=cpu, layer2=disk
    device_map = {
        "layer0": (torch.device("cuda"), torch.device("cuda")),  # device
        "layer1": (torch.device("cuda"), torch.device("cpu")),  # cpu
        "layer2": (torch.device("cuda"), "disk"),  # disk
    }
    dispatch_with_map(model, device_map, str(offload_dir))

    # Compress and decompress
    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)
    compressor.compress_model(model)
    dist.barrier()
    compressor.decompress_model(model)
    dist.barrier()

    # Weights should be back to float
    assert model.layer0.weight.dtype == torch.float32
    assert model.layer1.weight.dtype == torch.float32
    assert model.layer2.weight.dtype == torch.float32

    # Values should be close (within quantization error for 4-bit)
    diff0 = torch.abs(original_layer0 - model.layer0.weight.data)
    diff1 = torch.abs(original_layer1 - model.layer1.weight.data)
    diff2 = torch.abs(original_layer2 - model.layer2.weight.data)
    assert torch.max(diff0) < 1.0
    assert torch.max(diff1) < 1.0
    assert torch.max(diff2) < 1.0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_all_cpu_offload(tmp_path):
    """Test distributed compression with all layers offloaded to CPU."""
    # Create 3-layer model with W4A16 quantization
    model = ThreeLayerModel()
    setup_quantized_module(model.layer0, bits=4)
    setup_quantized_module(model.layer1, bits=4)
    setup_quantized_module(model.layer2, bits=4)

    # Setup all CPU offload
    device_map = {
        "layer0": (torch.device("cuda"), torch.device("cpu")),
        "layer1": (torch.device("cuda"), torch.device("cpu")),
        "layer2": (torch.device("cuda"), torch.device("cpu")),
    }
    dispatch_with_map(model, device_map, "")

    # Compress the model
    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)
    compressor.compress_model(model)
    dist.barrier()

    # Verify all layers are compressed
    assert hasattr(model.layer0, "weight_packed")
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model.layer2, "weight_packed")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_all_disk_offload(tmp_path):
    """Test distributed compression with all layers offloaded to disk."""
    # Create offload directory
    offload_dir = tmp_path / "offload_dir"
    os.makedirs(offload_dir, exist_ok=True)

    # Create 3-layer model with W4A16 quantization
    model = ThreeLayerModel()
    setup_quantized_module(model.layer0, bits=4)
    setup_quantized_module(model.layer1, bits=4)
    setup_quantized_module(model.layer2, bits=4)

    # Setup all disk offload
    device_map = {
        "layer0": (torch.device("cuda"), "disk"),
        "layer1": (torch.device("cuda"), "disk"),
        "layer2": (torch.device("cuda"), "disk"),
    }
    dispatch_with_map(model, device_map, str(offload_dir))

    # Compress the model
    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)
    compressor.compress_model(model)
    dist.barrier()

    # Verify all layers are compressed
    assert hasattr(model.layer0, "weight_packed")
    assert hasattr(model.layer1, "weight_packed")
    assert hasattr(model.layer2, "weight_packed")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_mixed_offload_load_balancing(tmp_path):
    """Test load balancing with mixed offload types across multiple layers."""
    # Create offload directory
    offload_dir = tmp_path / "offload_dir"
    os.makedirs(offload_dir, exist_ok=True)

    # Create model with 6 layers
    class SixLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(10, 10, bias=False)
            self.layer1 = nn.Linear(10, 10, bias=False)
            self.layer2 = nn.Linear(10, 10, bias=False)
            self.layer3 = nn.Linear(10, 10, bias=False)
            self.layer4 = nn.Linear(10, 10, bias=False)
            self.layer5 = nn.Linear(10, 10, bias=False)

    model = SixLayerModel()

    # Setup W4A16 quantization for all layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            setup_quantized_module(module, bits=4)

    # Setup mixed offload pattern: device, cpu, disk, device, cpu, disk
    device_map = {
        "layer0": (torch.device("cuda"), torch.device("cuda")),
        "layer1": (torch.device("cuda"), torch.device("cpu")),
        "layer2": (torch.device("cuda"), "disk"),
        "layer3": (torch.device("cuda"), torch.device("cuda")),
        "layer4": (torch.device("cuda"), torch.device("cpu")),
        "layer5": (torch.device("cuda"), "disk"),
    }
    dispatch_with_map(model, device_map, str(offload_dir))

    # Compress the model
    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)
    compressor.compress_model(model)
    dist.barrier()

    # Verify all layers are compressed
    for i in range(6):
        layer = getattr(model, f"layer{i}")
        assert hasattr(layer, "weight_packed"), f"layer{i} not compressed"
        assert layer.weight_packed.dtype == torch.int32

    # Verify consistency across ranks for all layers
    checksums = [
        getattr(model, f"layer{i}").weight_packed.sum().item() for i in range(6)
    ]

    if dist.get_rank() == 0:
        gathered = [None] * dist.get_world_size()
        dist.gather_object(checksums, gathered, dst=0)

        # All ranks should have identical checksums for all layers
        for i in range(1, dist.get_world_size()):
            for layer_idx in range(6):
                assert (
                    gathered[i][layer_idx] == gathered[0][layer_idx]
                ), f"Layer{layer_idx} mismatch between rank {i} and rank 0"
    else:
        dist.gather_object(checksums, None, dst=0)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_compression_mixed_offload_forward_pass(tmp_path):
    """Test forward pass after distributed compression with mixed offload."""
    # Create offload directory
    offload_dir = tmp_path / "offload_dir"
    os.makedirs(offload_dir, exist_ok=True)

    # Create 3-layer model with W4A16 quantization
    model = ThreeLayerModel()
    setup_quantized_module(model.layer0, bits=4)
    setup_quantized_module(model.layer1, bits=4)
    setup_quantized_module(model.layer2, bits=4)

    # Setup mixed offload: layer0=device, layer1=cpu, layer2=disk
    device_map = {
        "layer0": (torch.device("cuda"), torch.device("cuda")),  # device
        "layer1": (torch.device("cuda"), torch.device("cpu")),  # cpu
        "layer2": (torch.device("cuda"), "disk"),  # disk
    }
    dispatch_with_map(model, device_map, str(offload_dir))

    # Compress the model
    q_config = create_quantization_config(bits=4, format="pack-quantized")
    compressor = ModelCompressor(quantization_config=q_config)
    compressor.compress_model(model)
    dist.barrier()

    # Verify compressed
    assert hasattr(model.layer0, "weight_packed")
    assert hasattr(model, "ct_decompress_hook")

    # Forward pass should trigger decompression
    x = torch.randn(2, 10).cuda()
    _ = model(x)
    dist.barrier()

    # After forward, should be decompressed
    assert model.layer0.weight.dtype == torch.float32
    assert not hasattr(model.layer0, "weight_packed")
    assert not hasattr(model, "ct_decompress_hook")
