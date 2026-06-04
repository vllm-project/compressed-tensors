# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test that all converter validate() implementations work with tensors on meta device.
Meta device tensors have shape and dtype but no actual data, which is useful for
validating model structure without loading full weights into memory.
"""

import pytest
import torch
from compressed_tensors.entrypoints.convert import (
    AutoAWQConverter,
    CompressedTensorsDequantizer,
    FP8BlockDequantizer,
    ModelOptNvfp4Converter,
)


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Helper to pack int4 values for AWQ format"""
    values = values.to(torch.int32)
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int32)
    for offset in range(8):
        packed |= values[:, offset::8] << (offset * 4)
    return packed


@pytest.mark.unit
@pytest.mark.parametrize("zero_point", [True, False])
def test_autoawq_converter_validate_meta_device(zero_point):
    """Test AutoAWQConverter.validate() works with meta device tensors"""
    converter = AutoAWQConverter(
        group_size=2,
        targets=[r"re:.*proj$"],
        zero_point=zero_point,
    )

    # Create tensors on meta device - only shape and dtype matter for validation
    tensors = {
        "model.layers.0.mlp.up_proj.qweight": torch.empty(
            2, 1, dtype=torch.int32, device="meta"
        ),
        "model.layers.0.mlp.up_proj.scales": torch.empty(
            1, 8, dtype=torch.float16, device="meta"
        ),
        "model.embed_tokens.weight": torch.empty(
            4, 4, dtype=torch.float32, device="meta"
        ),
    }

    if zero_point:
        tensors["model.layers.0.mlp.up_proj.qzeros"] = torch.empty(
            1, 1, dtype=torch.int32, device="meta"
        )

    # Should not raise - validation only checks tensor presence and naming
    converter.validate(tensors)


@pytest.mark.unit
def test_autoawq_converter_validate_meta_device_missing_dependency():
    """Test AutoAWQConverter.validate() raises error for missing dependencies on meta device"""
    converter = AutoAWQConverter(zero_point=True)

    # Missing qzeros and scales
    tensors = {
        "model.layers.0.mlp.down_proj.qweight": torch.empty(
            1, 1, dtype=torch.int32, device="meta"
        )
    }

    with pytest.raises(ValueError, match="without corresponding"):
        converter.validate(tensors)


@pytest.mark.unit
def test_autoawq_converter_validate_meta_device_non_targeted():
    """Test AutoAWQConverter.validate() rejects non-targeted tensors on meta device"""
    converter = AutoAWQConverter(targets=[r"re:.*up_proj$"])

    # down_proj is not targeted but has AWQ tensors
    tensors = {
        "model.layers.0.mlp.down_proj.qweight": torch.empty(
            1, 1, dtype=torch.int32, device="meta"
        )
    }

    with pytest.raises(ValueError, match="unexpected non-targeted tensor"):
        converter.validate(tensors)


@pytest.mark.unit
def test_fp8block_converter_validate_meta_device():
    """Test FP8BlockDequantizer.validate() works with meta device tensors"""
    converter = FP8BlockDequantizer(
        targets=[r"re:.*layer\d+\.mlp\..*proj$"],
        weight_block_size=(128, 128),
    )

    tensors = {
        "model.layer0.mlp.up_proj.weight": torch.empty(
            256, 256, dtype=torch.float8_e4m3fn, device="meta"
        ),
        "model.layer0.mlp.up_proj.weight_scale_inv": torch.empty(
            2, 2, dtype=torch.float32, device="meta"
        ),
        "model.layer1.mlp.down_proj.weight": torch.empty(
            256, 256, dtype=torch.float8_e4m3fn, device="meta"
        ),
        "model.layer1.mlp.down_proj.weight_scale_inv": torch.empty(
            2, 2, dtype=torch.float32, device="meta"
        ),
        "model.embed_tokens.weight": torch.empty(
            128, 128, dtype=torch.bfloat16, device="meta"
        ),
    }

    # Should not raise
    converter.validate(tensors)


@pytest.mark.unit
def test_fp8block_converter_validate_meta_device_missing_scale():
    """Test FP8BlockDequantizer.validate() raises error for missing scale_inv on meta device"""
    converter = FP8BlockDequantizer(targets=[r"re:.*layer\d+\.mlp\..*proj$"])

    # Missing weight_scale_inv
    tensors = {
        "model.layer0.mlp.up_proj.weight": torch.empty(
            256, 256, dtype=torch.float8_e4m3fn, device="meta"
        ),
    }

    with pytest.raises(ValueError, match="without corresponding weight_scale_inv"):
        converter.validate(tensors)


@pytest.mark.unit
def test_fp8block_converter_validate_meta_device_missing_weight():
    """Test FP8BlockDequantizer.validate() raises error for missing weight on meta device"""
    converter = FP8BlockDequantizer(targets=[r"re:.*layer\d+\.mlp\..*proj$"])

    # Missing weight
    tensors = {
        "model.layer0.mlp.up_proj.weight_scale_inv": torch.empty(
            2, 2, dtype=torch.float32, device="meta"
        ),
    }

    with pytest.raises(ValueError, match="without corresponding weight"):
        converter.validate(tensors)


@pytest.mark.unit
def test_fp8block_converter_validate_meta_device_disallowed_untargeted():
    """Test FP8BlockDequantizer.validate() rejects untargeted scale_inv on meta device"""
    converter = FP8BlockDequantizer(targets=[r"re:.*layer\d+\.mlp\..*proj$"])

    # Untargeted module with weight_scale_inv
    tensors = {
        "model.embed_tokens.weight_scale_inv": torch.empty(
            2, 2, dtype=torch.float32, device="meta"
        ),
    }

    with pytest.raises(ValueError, match="unexpected non-targeted tensor"):
        converter.validate(tensors)


@pytest.mark.unit
def test_modelopt_nvfp4_converter_validate_meta_device():
    """Test ModelOptNvfp4Converter.validate() works with meta device tensors"""
    converter = ModelOptNvfp4Converter(targets=[r"re:.*layer\d+\.mlp\..*proj$"])

    tensors = {
        "model.layer0.mlp.up_proj.input_scale": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
        "model.layer0.mlp.up_proj.weight": torch.empty(
            256, 256, dtype=torch.uint8, device="meta"
        ),
        "model.layer0.mlp.up_proj.weight_scale": torch.empty(
            2, 2, dtype=torch.float8_e4m3fn, device="meta"
        ),
        "model.layer0.mlp.up_proj.weight_scale_2": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
        "model.embed_tokens.weight": torch.empty(
            128, 128, dtype=torch.bfloat16, device="meta"
        ),
    }

    # Should not raise
    converter.validate(tensors)


@pytest.mark.unit
def test_modelopt_nvfp4_converter_validate_meta_device_with_kv_cache():
    """Test ModelOptNvfp4Converter.validate() works with kv_cache on meta device"""
    from compressed_tensors.quantization import QuantizationArgs

    kv_cache_scheme = QuantizationArgs(num_bits=8, type="int", symmetric=True)
    converter = ModelOptNvfp4Converter(
        targets=[r"re:.*layer\d+\.self_attn\..*proj$"],
        kv_cache_scheme=kv_cache_scheme,
    )

    tensors = {
        "model.layer0.self_attn.k_proj.input_scale": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
        "model.layer0.self_attn.k_proj.weight": torch.empty(
            256, 256, dtype=torch.uint8, device="meta"
        ),
        "model.layer0.self_attn.k_proj.weight_scale": torch.empty(
            2, 2, dtype=torch.float8_e4m3fn, device="meta"
        ),
        "model.layer0.self_attn.k_proj.weight_scale_2": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
        "model.layer0.self_attn.k_proj.k_scale": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
        "model.layer0.self_attn.v_proj.input_scale": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
        "model.layer0.self_attn.v_proj.weight": torch.empty(
            256, 256, dtype=torch.uint8, device="meta"
        ),
        "model.layer0.self_attn.v_proj.weight_scale": torch.empty(
            2, 2, dtype=torch.float8_e4m3fn, device="meta"
        ),
        "model.layer0.self_attn.v_proj.weight_scale_2": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
        "model.layer0.self_attn.v_proj.v_scale": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
    }

    # Should not raise
    converter.validate(tensors)


@pytest.mark.unit
def test_modelopt_nvfp4_converter_validate_meta_device_disallowed_untargeted():
    """Test ModelOptNvfp4Converter.validate() rejects untargeted modelopt params on meta device"""
    converter = ModelOptNvfp4Converter(targets=[r"re:.*layer\d+\.mlp\..*proj$"])

    # Untargeted module with modelopt params
    tensors = {
        "model.embed_tokens.input_scale": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
    }

    with pytest.raises(ValueError, match="unexpected non-targeted tensor"):
        converter.validate(tensors)


@pytest.mark.unit
def test_ct_dequantizer_validate_meta_device(tmp_path):
    """Test CompressedTensorsDequantizer.validate() works with meta device tensors"""
    # Create a mock config.json with quantization config
    config_path = tmp_path / "config.json"
    config_content = """{
        "model_type": "llama",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": true,
                        "group_size": 128,
                        "strategy": "group"
                    }
                }
            }
        }
    }"""
    config_path.write_text(config_content)

    # Create a mock weight file
    weight_file = tmp_path / "model.safetensors"
    weight_file.write_bytes(b"")  # Empty file, just needs to exist

    converter = CompressedTensorsDequantizer(tmp_path)

    # Create meta device tensors matching pack-quantized format
    tensors = {
        "model.layers.0.self_attn.q_proj.weight_packed": torch.empty(
            128, 32, dtype=torch.int32, device="meta"
        ),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.empty(
            128, 4, dtype=torch.float16, device="meta"
        ),
        "model.layers.0.self_attn.q_proj.weight_shape": torch.empty(
            2, dtype=torch.int64, device="meta"
        ),
        "model.embed_tokens.weight": torch.empty(
            1024, 512, dtype=torch.float16, device="meta"
        ),
    }

    # Should not raise - validation only checks tensor presence
    converter.validate(tensors)


@pytest.mark.unit
def test_ct_dequantizer_validate_meta_device_missing_param(tmp_path):
    """Test CompressedTensorsDequantizer.validate() raises error for missing params on meta device"""
    # Create a mock config.json with quantization config
    config_path = tmp_path / "config.json"
    config_content = """{
        "model_type": "llama",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": true,
                        "group_size": 128,
                        "strategy": "group"
                    }
                }
            }
        }
    }"""
    config_path.write_text(config_content)

    # Create a mock weight file
    weight_file = tmp_path / "model.safetensors"
    weight_file.write_bytes(b"")

    converter = CompressedTensorsDequantizer(tmp_path)

    # Missing weight_scale
    tensors = {
        "model.layers.0.self_attn.q_proj.weight_packed": torch.empty(
            128, 32, dtype=torch.int32, device="meta"
        ),
        "model.layers.0.self_attn.q_proj.weight_shape": torch.empty(
            2, dtype=torch.int64, device="meta"
        ),
    }

    with pytest.raises(ValueError, match="Expected key .* not found"):
        converter.validate(tensors)


@pytest.mark.unit
def test_ct_dequantizer_validate_meta_device_unconsumed_keys(tmp_path):
    """Test CompressedTensorsDequantizer.validate() raises error for unconsumed keys on meta device"""
    # Create a mock config.json with quantization config
    config_path = tmp_path / "config.json"
    config_content = """{
        "model_type": "llama",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": true,
                        "group_size": 128,
                        "strategy": "group"
                    }
                }
            }
        }
    }"""
    config_path.write_text(config_content)

    # Create a mock weight file
    weight_file = tmp_path / "model.safetensors"
    weight_file.write_bytes(b"")

    converter = CompressedTensorsDequantizer(tmp_path)

    # Extra unexpected key for a matched module
    tensors = {
        "model.layers.0.self_attn.q_proj.weight_packed": torch.empty(
            128, 32, dtype=torch.int32, device="meta"
        ),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.empty(
            128, 4, dtype=torch.float16, device="meta"
        ),
        "model.layers.0.self_attn.q_proj.weight_shape": torch.empty(
            2, dtype=torch.int64, device="meta"
        ),
        "model.layers.0.self_attn.q_proj.unexpected_param": torch.empty(
            1, dtype=torch.float32, device="meta"
        ),
    }

    with pytest.raises(ValueError, match="unconsumed keys"):
        converter.validate(tensors)
