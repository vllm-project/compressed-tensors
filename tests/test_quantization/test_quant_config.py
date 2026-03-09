# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_FORMAT,
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from pydantic import ValidationError


def test_basic_config():
    config_groups = {"group_1": QuantizationScheme(targets=[])}
    config = QuantizationConfig(config_groups=config_groups)

    assert config.config_groups == config_groups
    assert config.quant_method == DEFAULT_QUANTIZATION_METHOD
    assert config.format == DEFAULT_QUANTIZATION_FORMAT
    assert config.quantization_status == QuantizationStatus.INITIALIZED
    assert config.global_compression_ratio is None
    assert isinstance(config.ignore, list) and len(config.ignore) == 0


def test_full_config():
    config_groups = {
        "group_1": QuantizationScheme(targets=[]),
        "group_2": QuantizationScheme(targets=[]),
    }
    global_compression_ratio = 3.5
    ignore = ["model.layers.0"]
    quantization_status = "compressed"

    config = QuantizationConfig(
        config_groups=config_groups,
        global_compression_ratio=global_compression_ratio,
        ignore=ignore,
        quantization_status=quantization_status,
    )
    assert config.config_groups == config_groups
    assert config.global_compression_ratio == global_compression_ratio
    assert config.ignore == ignore
    assert config.quantization_status == QuantizationStatus.COMPRESSED


def test_need_config_groups():
    with pytest.raises(ValidationError):
        _ = QuantizationScheme()


@pytest.mark.parametrize(
    "scheme_name",
    ["W8A8", "W8A16", "W4A16", "FP8"],
)
def test_load_scheme_from_preset(scheme_name: str):
    targets = ["Linear"]
    config = QuantizationConfig(config_groups={scheme_name: targets})

    assert scheme_name in config.config_groups
    assert isinstance(config.config_groups[scheme_name], QuantizationScheme)
    assert config.config_groups[scheme_name].targets == targets


def test_to_dict():
    """Test serialization of QuantizationConfig including format"""
    from compressed_tensors.quantization import QuantizationArgs

    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, symmetric=True, group_size=128),
        ),
        "group_2": QuantizationScheme(
            targets=["Conv2d"],
            weights=QuantizationArgs(num_bits=8),
        ),
    }
    config = QuantizationConfig(
        config_groups=config_groups,
        global_compression_ratio=3.5,
        ignore=["model.layers.0"],
        quantization_status="compressed",
        format="int-quantized",
    )

    # Serialize to dict
    config_dict = config.to_dict()
    assert "config_groups" in config_dict
    assert config_dict["format"] == "int-quantized"
    assert config_dict["quantization_status"] == "compressed"

    # Deserialize from dict
    reloaded = QuantizationConfig.model_validate(config_dict)
    assert config == reloaded
