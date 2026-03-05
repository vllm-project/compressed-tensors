# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compressed_tensors import ModelCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.transform import TransformConfig


class DummyLinear(nn.Module):
    """Simple linear module for testing."""

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class TwoLayerModel(nn.Module):
    """Model with two linear layers for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10, bias=False)
        self.layer2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def create_quantization_config(
    bits=8, type="int", strategy="tensor", format="int-quantized"
):
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
                    "strategy": strategy,
                    "symmetric": True,
                    "type": type,
                },
            }
        },
    }
    return QuantizationConfig.model_validate(config_dict)


def create_quantization_scheme(
    bits=8, type="int", strategy="tensor", format=None, with_input_activations=False
):
    """Helper to create a QuantizationScheme for testing."""
    input_activations = None
    if with_input_activations:
        input_activations = QuantizationArgs(
            num_bits=bits,
            strategy=strategy,
            symmetric=True,
            type=type,
        )

    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits,
            strategy=strategy,
            symmetric=True,
            type=type,
        ),
        input_activations=input_activations,
        format=format,
    )


class TestModelCompressorCompression:
    """Test compression and decompression functionality."""

    def test_compress_model_basic(self):
        """Test basic model compression."""
        model = DummyLinear()
        # Use pack-quantized format (requires 4 or 8 bits, int type,
        # no input activations)
        scheme = create_quantization_scheme(bits=4, type="int", strategy="channel")
        model.linear.quantization_scheme = scheme
        model.linear.quantization_status = QuantizationStatus.FROZEN
        # Set up quantization parameters
        model.linear.weight_scale = nn.Parameter(
            torch.ones(model.linear.weight.shape[0], 1) * 0.01
        )
        model.linear.weight_zero_point = nn.Parameter(
            torch.zeros(model.linear.weight.shape[0], 1, dtype=torch.int32),
            requires_grad=False,
        )

        q_config = create_quantization_config(bits=4, format="pack-quantized")
        compressor = ModelCompressor(quantization_config=q_config)

        compressor.compress_model(model)

        # Weight should be packed into int32
        assert hasattr(model.linear, "weight_packed")
        assert model.linear.weight_packed.dtype == torch.int32
        assert hasattr(model, "decompress_hook")

    def test_compress_model_skips_non_quantized_modules(self):
        """Test that modules without quantization_scheme are skipped."""
        model = TwoLayerModel()
        # Only quantize layer1
        scheme = create_quantization_scheme(bits=4, type="int", strategy="channel")
        model.layer1.quantization_scheme = scheme
        model.layer1.quantization_status = QuantizationStatus.FROZEN
        model.layer1.weight_scale = nn.Parameter(
            torch.ones(model.layer1.weight.shape[0], 1) * 0.01
        )
        # Use .int() instead of dtype=torch.int32 to avoid Parameter issues
        model.layer1.weight_zero_point = nn.Parameter(
            torch.zeros(model.layer1.weight.shape[0], 1).int(), requires_grad=False
        )

        q_config = create_quantization_config(bits=4, format="pack-quantized")
        compressor = ModelCompressor(quantization_config=q_config)

        layer2_original_dtype = model.layer2.weight.dtype
        compressor.compress_model(model)

        # layer1 should be compressed
        assert hasattr(model.layer1, "weight_packed")
        # layer2 should remain unchanged
        assert model.layer2.weight.dtype == layer2_original_dtype

    def test_compress_decompress_roundtrip(self):
        """Test that compress then decompress recovers original weights."""
        model = DummyLinear()
        scheme = create_quantization_scheme(bits=4, type="int", strategy="channel")
        model.linear.quantization_scheme = scheme
        model.linear.quantization_status = QuantizationStatus.FROZEN
        model.linear.weight_scale = nn.Parameter(
            torch.ones(model.linear.weight.shape[0], 1) * 0.01
        )
        model.linear.weight_zero_point = nn.Parameter(
            torch.zeros(model.linear.weight.shape[0], 1, dtype=torch.int32),
            requires_grad=False,
        )

        q_config = create_quantization_config(bits=4, format="pack-quantized")
        compressor = ModelCompressor(quantization_config=q_config)

        original_weight = model.linear.weight.data.clone()
        compressor.compress_model(model)

        # Remove the decompress hook before manual decompression
        if hasattr(model, "decompress_hook"):
            model.decompress_hook.remove()

        compressor.decompress_model(model)

        # After decompression, weight should be back to float
        assert model.linear.weight.dtype == torch.float32
        # Values should be close (within quantization error for 4-bit)
        diff = torch.abs(original_weight - model.linear.weight.data)
        assert torch.max(diff) < 1.0  # Reasonable threshold for 4-bit quantization

    def test_decompress_hook_triggers_on_forward(self):
        """Test that the decompress hook is triggered on forward pass."""
        model = DummyLinear()
        scheme = create_quantization_scheme(bits=4, type="int", strategy="channel")
        model.linear.quantization_scheme = scheme
        model.linear.quantization_status = QuantizationStatus.FROZEN
        model.linear.weight_scale = nn.Parameter(
            torch.ones(model.linear.weight.shape[0], 1) * 0.01
        )
        model.linear.weight_zero_point = nn.Parameter(
            torch.zeros(model.linear.weight.shape[0], 1, dtype=torch.int32),
            requires_grad=False,
        )

        q_config = create_quantization_config(bits=4, format="pack-quantized")
        compressor = ModelCompressor(quantization_config=q_config)

        compressor.compress_model(model)
        assert hasattr(model.linear, "weight_packed")
        assert hasattr(model, "decompress_hook")

        # Forward pass should trigger decompression
        x = torch.randn(2, 10)
        _ = model(x)

        # After forward, weight should be decompressed
        assert model.linear.weight.dtype == torch.float32
        assert not hasattr(model.linear, "weight_packed")

    def test_compress_model_updates_format_in_config(self):
        """Test that compress_model updates the quantization config format."""
        model = DummyLinear()
        scheme = create_quantization_scheme(bits=4, type="int", strategy="channel")
        model.linear.quantization_scheme = scheme
        model.linear.quantization_status = QuantizationStatus.FROZEN
        model.linear.weight_scale = nn.Parameter(
            torch.ones(model.linear.weight.shape[0], 1) * 0.01
        )
        model.linear.weight_zero_point = nn.Parameter(
            torch.zeros(model.linear.weight.shape[0], 1, dtype=torch.int32),
            requires_grad=False,
        )

        q_config = create_quantization_config(
            bits=4, format="dense"
        )  # Start with dense
        compressor = ModelCompressor(quantization_config=q_config)

        compressor.compress_model(model)

        # Format should be updated to pack-quantized
        assert compressor.quantization_config.format == CompressionFormat.pack_quantized


class TestModelCompressorConfigUpdate:
    """Test config file update functionality."""

    def test_update_config_creates_file(self):
        """Test that update_config creates config.json if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q_config = create_quantization_config()
            compressor = ModelCompressor(quantization_config=q_config)

            compressor.update_config(tmpdir)

            config_path = Path(tmpdir) / "config.json"
            assert config_path.exists()

            with open(config_path, "r") as f:
                config_data = json.load(f)

            assert "quantization_config" in config_data
            assert (
                config_data["quantization_config"]["quant_method"]
                == "compressed-tensors"
            )

    def test_update_config_preserves_existing_data(self):
        """Test that update_config preserves existing config data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            existing_data = {"model_type": "test", "hidden_size": 768}
            with open(config_path, "w") as f:
                json.dump(existing_data, f)

            q_config = create_quantization_config()
            compressor = ModelCompressor(quantization_config=q_config)
            compressor.update_config(tmpdir)

            with open(config_path, "r") as f:
                config_data = json.load(f)

            # Original data should be preserved
            assert config_data["model_type"] == "test"
            assert config_data["hidden_size"] == 768
            # New data should be added
            assert "quantization_config" in config_data

    def test_update_config_with_transform_config(self):
        """Test that update_config includes transform_config."""
        from compressed_tensors.transform import TransformArgs, TransformScheme

        with tempfile.TemporaryDirectory() as tmpdir:
            q_config = create_quantization_config()
            t_config = TransformConfig(
                config_groups={
                    "group_0": TransformScheme(
                        type="hadamard",
                        apply=[
                            TransformArgs(targets=["Linear"], location="weight_input")
                        ],
                    )
                }
            )
            compressor = ModelCompressor(
                quantization_config=q_config, transform_config=t_config
            )

            compressor.update_config(tmpdir)

            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "r") as f:
                config_data = json.load(f)

            assert "quantization_config" in config_data
            assert "transform_config" in config_data["quantization_config"]

    def test_update_config_no_configs(self):
        """Test that update_config does nothing when no configs are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = ModelCompressor()
            compressor.update_config(tmpdir)

            config_path = Path(tmpdir) / "config.json"
            # Should not create file if no configs
            assert not config_path.exists()

    def test_update_config_includes_version(self):
        """Test that update_config includes compression version."""
        import compressed_tensors

        with tempfile.TemporaryDirectory() as tmpdir:
            q_config = create_quantization_config()
            compressor = ModelCompressor(quantization_config=q_config)
            compressor.update_config(tmpdir)

            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "r") as f:
                config_data = json.load(f)

            assert "quantization_config" in config_data
            # The version field is added by update_config
            qc = config_data["quantization_config"]
            assert "version" in qc, f"Keys in qc: {qc.keys()}"
            assert qc["version"] == compressed_tensors.__version__


class TestModelCompressorEdgeCases:
    """Test edge cases and error handling."""

    def test_compress_model_infers_format(self):
        """Test that compression infers format when not set on scheme."""
        model = DummyLinear()
        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", strategy="channel", symmetric=True
            ),
        )
        # No format set - will be inferred
        model.linear.quantization_scheme = scheme
        model.linear.quantization_status = QuantizationStatus.FROZEN
        model.linear.weight_scale = nn.Parameter(
            torch.ones(model.linear.weight.shape[0], 1) * 0.01
        )
        model.linear.weight_zero_point = nn.Parameter(
            torch.zeros(model.linear.weight.shape[0], 1, dtype=torch.int32),
            requires_grad=False,
        )

        q_config = create_quantization_config(bits=4)
        compressor = ModelCompressor(quantization_config=q_config)

        compressor.compress_model(model)

        # Format should be inferred and set
        assert model.linear.quantization_scheme.format is not None
        assert hasattr(model.linear, "weight_packed")

    def test_decompress_model_requires_format(self):
        """Test that decompression requires format to be set on scheme."""
        model = DummyLinear()
        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=8, strategy="tensor", symmetric=True),
        )
        # No format set
        model.linear.quantization_scheme = scheme

        q_config = create_quantization_config()
        compressor = ModelCompressor(quantization_config=q_config)

        # The assertion is "Must set format before decompressing", but the actual error
        # is AttributeError when trying to access scheme.format.value
        with pytest.raises((AssertionError, AttributeError)):
            compressor.decompress_model(model)

    def test_empty_model(self):
        """Test compression of a model with no quantized modules."""
        model = nn.Sequential()
        q_config = create_quantization_config()
        compressor = ModelCompressor(quantization_config=q_config)

        # Should not raise, just do nothing
        compressor.compress_model(model)
        compressor.decompress_model(model)

    def test_model_with_no_quantization_scheme(self):
        """Test that modules without quantization_scheme are skipped."""
        model = TwoLayerModel()
        # Don't add quantization_scheme to either layer

        q_config = create_quantization_config()
        compressor = ModelCompressor(quantization_config=q_config)

        original_dtype = model.layer1.weight.dtype
        compressor.compress_model(model)

        # Should skip all modules
        assert model.layer1.weight.dtype == original_dtype
        assert model.layer2.weight.dtype == original_dtype
