# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme


def test_compression_format_serializable():
    """Test that CompressionFormat can be serialized to JSON"""
    format = CompressionFormat.int_quantized

    # Test direct JSON serialization
    json_str = json.dumps(format)
    assert json_str == '"int-quantized"'

    # Test deserialization
    deserialized = CompressionFormat(json.loads(json_str))
    assert deserialized == format


def test_compression_format_all_values():
    """Test that all CompressionFormat values are serializable"""
    for format in CompressionFormat:
        # Serialize to JSON
        json_str = json.dumps(format)
        assert isinstance(json_str, str)

        # Deserialize from JSON
        deserialized = CompressionFormat(json.loads(json_str))
        assert deserialized == format


def test_compression_format_in_dict():
    """Test that CompressionFormat can be serialized in a dict"""
    test_dict = {
        "format": CompressionFormat.pack_quantized,
        "other_field": "value",
    }

    # Serialize to JSON
    json_str = json.dumps(test_dict, default=str)
    parsed = json.loads(json_str)

    assert parsed["format"] == "pack-quantized"
    assert parsed["other_field"] == "value"


def test_compression_format_in_scheme():
    """Test that CompressionFormat serializes properly in QuantizationScheme"""
    scheme = QuantizationScheme(
        targets=["Linear"], format=CompressionFormat.int_quantized
    )

    # Serialize to dict
    scheme_dict = scheme.model_dump()
    assert scheme_dict["format"] == "int-quantized"
    assert isinstance(scheme_dict["format"], str)

    # Serialize to JSON
    json_str = json.dumps(scheme_dict)
    parsed = json.loads(json_str)
    assert parsed["format"] == "int-quantized"

    # Deserialize from dict
    reloaded = QuantizationScheme.model_validate(parsed)
    assert reloaded.format == CompressionFormat.int_quantized


def test_compression_format_in_config():
    """Test that CompressionFormat serializes properly in QuantizationConfig"""
    config = QuantizationConfig(
        config_groups={"group_1": QuantizationScheme(targets=[])},
        format=CompressionFormat.float_quantized.value,
    )

    # Serialize to dict
    config_dict = config.to_dict()
    assert config_dict["format"] == "float-quantized"
    assert isinstance(config_dict["format"], str)

    # Serialize to JSON
    json_str = json.dumps(config_dict)
    parsed = json.loads(json_str)
    assert parsed["format"] == "float-quantized"

    # Deserialize from dict
    reloaded = QuantizationConfig.model_validate(parsed)
    assert reloaded.format == "float-quantized"


@pytest.mark.parametrize(
    "format_value",
    [
        CompressionFormat.dense,
        CompressionFormat.int_quantized,
        CompressionFormat.float_quantized,
        CompressionFormat.pack_quantized,
        CompressionFormat.naive_quantized,
        CompressionFormat.mixed_precision,
        CompressionFormat.nvfp4_pack_quantized,
        CompressionFormat.mxfp4_pack_quantized,
    ],
)
def test_compression_format_round_trip(format_value):
    """Test round-trip serialization for each CompressionFormat value"""
    # Create a config with the format
    config = QuantizationConfig(
        config_groups={"group_1": QuantizationScheme(targets=["Linear"])},
        format=format_value.value,
    )

    # Serialize to dict then JSON
    config_dict = config.to_dict()
    json_str = json.dumps(config_dict)

    # Deserialize from JSON
    parsed = json.loads(json_str)
    reloaded = QuantizationConfig.model_validate(parsed)

    # Verify format is preserved
    assert reloaded.format == format_value.value
    assert config == reloaded
