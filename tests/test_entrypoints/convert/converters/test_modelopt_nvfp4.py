# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert import ModelOptNvfp4Converter
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)


@pytest.mark.unit
def test_modelopt_nvfp4_converter_process():
    """
    Test that the converter's process method correctly transforms ModelOpt NVFP4
    tensors to compressed-tensors NVFP4 format.
    """
    converter = ModelOptNvfp4Converter(targets=[r"re:.*layer\d+\.mlp\..*proj$"])

    # Create mock tensors dict with ModelOpt NVFP4 format
    input_scale = torch.tensor([2.0], dtype=torch.float32)
    weight = torch.randint(0, 255, (256, 256), dtype=torch.uint8)
    weight_scale = torch.rand(256, 1, dtype=torch.float32).to(torch.float8_e4m3fn)
    weight_scale_2 = torch.tensor([4.0], dtype=torch.float32)
    embed_weight = torch.randn(128, 128, dtype=torch.bfloat16)

    tensors = {
        "model.layer0.mlp.up_proj.input_scale": input_scale,
        "model.layer0.mlp.up_proj.weight": weight,
        "model.layer0.mlp.up_proj.weight_scale": weight_scale,
        "model.layer0.mlp.up_proj.weight_scale_2": weight_scale_2,
        "model.embed_tokens.weight": embed_weight,
    }

    # Process the tensors
    result = converter.process(tensors)

    # Verify transformations
    # input_scale -> input_global_scale (inverted)
    assert "model.layer0.mlp.up_proj.input_scale" not in result
    assert "model.layer0.mlp.up_proj.input_global_scale" in result
    assert torch.allclose(
        result["model.layer0.mlp.up_proj.input_global_scale"],
        1 / input_scale,
    )

    # weight -> weight_packed (renamed)
    assert "model.layer0.mlp.up_proj.weight" not in result
    assert "model.layer0.mlp.up_proj.weight_packed" in result
    assert torch.equal(result["model.layer0.mlp.up_proj.weight_packed"], weight)

    # weight_scale stays the same
    assert "model.layer0.mlp.up_proj.weight_scale" in result
    assert (
        result["model.layer0.mlp.up_proj.weight_scale"].data_ptr()
        == weight_scale.data_ptr()
    )

    # weight_scale_2 -> weight_global_scale (inverted)
    assert "model.layer0.mlp.up_proj.weight_scale_2" not in result
    assert "model.layer0.mlp.up_proj.weight_global_scale" in result
    assert torch.allclose(
        result["model.layer0.mlp.up_proj.weight_global_scale"],
        1 / weight_scale_2,
    )

    # Non-targeted tensor should not be modified
    assert torch.equal(result["model.embed_tokens.weight"], embed_weight)


@pytest.mark.unit
def test_modelopt_nvfp4_converter_get_dependencies():
    """
    Test that get_dependencies returns the correct dependent tensors for
    targeted weight tensors.
    """
    converter = ModelOptNvfp4Converter(targets=[r"re:.*down_proj$"])

    # Targeted layer should have dependencies
    deps = converter.get_dependencies("model.layer0.mlp.down_proj.weight")
    assert deps == {
        "model.layer0.mlp.down_proj.input_scale",
        "model.layer0.mlp.down_proj.weight_scale",
        "model.layer0.mlp.down_proj.weight_scale_2",
    }

    # Non-targeted layer should have no dependencies
    deps = converter.get_dependencies("model.layer0.mlp.up_proj.weight")
    assert deps == set()

    # Non-weight tensor should have no dependencies
    deps = converter.get_dependencies("model.layer0.mlp.down_proj.weight_scale")
    assert deps == set()


@pytest.mark.unit
def test_modelopt_nvfp4_converter_validate_with_meta_tensors():
    """
    Test that the converter's validate method works correctly with meta tensors
    and returns the correct output dict.
    """
    converter = ModelOptNvfp4Converter(targets=[r"re:.*layer\d+\.mlp\..*proj$"])

    # Create mock tensors dict with NVFP4 tensors on meta device
    with torch.device("meta"):
        tensors = {
            "model.layer0.mlp.up_proj.input_scale": torch.empty(1, dtype=torch.float32),
            "model.layer0.mlp.up_proj.weight": torch.empty(256, 256, dtype=torch.uint8),
            "model.layer0.mlp.up_proj.weight_scale": torch.empty(
                256, 1, dtype=torch.float8_e4m3fn
            ),
            "model.layer0.mlp.up_proj.weight_scale_2": torch.empty(
                1, dtype=torch.float32
            ),
            "model.layer1.mlp.down_proj.input_scale": torch.empty(
                1, dtype=torch.float32
            ),
            "model.layer1.mlp.down_proj.weight": torch.empty(
                256, 256, dtype=torch.uint8
            ),
            "model.layer1.mlp.down_proj.weight_scale": torch.empty(
                256, 1, dtype=torch.float8_e4m3fn
            ),
            "model.layer1.mlp.down_proj.weight_scale_2": torch.empty(
                1, dtype=torch.float32
            ),
            "model.embed_tokens.weight": torch.empty(128, 128, dtype=torch.bfloat16),
        }

    result = converter.validate(tensors)

    # Renamed params present with correct dtypes
    assert "model.layer0.mlp.up_proj.input_global_scale" in result
    assert result["model.layer0.mlp.up_proj.input_global_scale"].dtype == torch.float32
    assert "model.layer0.mlp.up_proj.weight_packed" in result
    assert result["model.layer0.mlp.up_proj.weight_packed"].dtype == torch.uint8
    assert "model.layer0.mlp.up_proj.weight_global_scale" in result
    assert result["model.layer0.mlp.up_proj.weight_global_scale"].dtype == torch.float32

    # weight_scale stays (no rename)
    assert "model.layer0.mlp.up_proj.weight_scale" in result

    # Source names removed
    assert "model.layer0.mlp.up_proj.input_scale" not in result
    assert "model.layer0.mlp.up_proj.weight" not in result
    assert "model.layer0.mlp.up_proj.weight_scale_2" not in result

    # Untargeted passes through
    assert "model.embed_tokens.weight" in result


@pytest.mark.unit
def test_modelopt_nvfp4_update_config_from_none():
    converter = ModelOptNvfp4Converter(targets=[r"re:.*proj$"])

    config = converter.update_config(None)

    assert config is not None
    assert len(config.config_groups) == 1
    assert "config_group_0" in config.config_groups
    assert config.format == CompressionFormat.nvfp4_pack_quantized.value
    assert config.quantization_status == QuantizationStatus.COMPRESSED


@pytest.mark.unit
def test_modelopt_nvfp4_update_config_merges():
    converter = ModelOptNvfp4Converter(targets=[r"re:.*proj$"])

    # Start with an existing config from another converter
    existing = QuantizationConfig(
        config_groups={
            "prior_group": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=8, type="int", symmetric=True),
            )
        },
    )

    merged = converter.update_config(existing)

    assert merged is existing  # in-place mutation
    assert len(merged.config_groups) == 2  # prior_group + config_group_0
    assert "prior_group" in merged.config_groups


@pytest.mark.unit
def test_modelopt_nvfp4_validate_raises_on_untargeted_qparam():
    """
    An untargeted module carrying a quantization param that should only appear on
    targeted modules must raise ValueError. Regression for validation dropped in
    the multi-converter refactor (#805).
    """
    converter = ModelOptNvfp4Converter(targets=[r"re:.*up_proj$"])

    with torch.device("meta"):
        tensors = {
            "model.layer0.mlp.up_proj.input_scale": torch.empty(1, dtype=torch.float32),
            "model.layer0.mlp.up_proj.weight": torch.empty(256, 256, dtype=torch.uint8),
            "model.layer0.mlp.up_proj.weight_scale": torch.empty(
                256, 1, dtype=torch.float8_e4m3fn
            ),
            "model.layer0.mlp.up_proj.weight_scale_2": torch.empty(
                1, dtype=torch.float32
            ),
            # untargeted module carrying a disallowed qparam
            "model.layer0.mlp.down_proj.weight_scale": torch.empty(
                256, 1, dtype=torch.float8_e4m3fn
            ),
        }

    with pytest.raises(ValueError):
        converter.validate(tensors)


@pytest.fixture
def modelopt_nvfp4_tensors():
    tensors = {
        "model.embed_tokens.weight": torch.ones(128, 64, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(128, 64, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(64, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(64, dtype=torch.bfloat16),
    }
    for module in ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"):
        tensors.update(
            {
                f"{module}.weight": torch.full((64, 32), 0x12, dtype=torch.uint8),
                f"{module}.weight_scale": torch.ones(64, 4).to(torch.float8_e4m3fn),
                f"{module}.weight_scale_2": torch.tensor(0.25),
                f"{module}.input_scale": torch.tensor(0.5),
            }
        )
    return tensors


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"targets": ["Linear"]}, id="linear"),
        pytest.param(
            {"targets": ["Linear"], "ignore": ["lm_head"]}, id="linear-ignore-lm-head"
        ),
        pytest.param(
            {
                "targets": [
                    r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
                    r"re:.*self_attn.*\.(q|k|v|o)_proj$",
                ],
                "kv_cache_scheme": QuantizationArgs(
                    num_bits=8, dynamic=False, type="float"
                ),
            },
            id="qwen3-example",
        ),
    ],
)
@pytest.mark.parametrize("method", ["process", "validate"])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_modelopt_nvfp4_preserves_unquantized_weights(
    modelopt_nvfp4_tensors, kwargs, method, device
):
    tensors = {
        name: tensor.to(device) for name, tensor in modelopt_nvfp4_tensors.items()
    }
    converter = ModelOptNvfp4Converter(**kwargs)

    result = getattr(converter, method)(dict(tensors))

    expected_names = set()
    for name, tensor in tensors.items():
        module, _, param = name.rpartition(".")
        if param == "weight" and tensor.dtype == torch.uint8:
            converted_name = f"{module}.weight_packed"
        elif param in {"input_scale", "weight_scale_2"}:
            scale = "input" if param == "input_scale" else "weight"
            converted_name = f"{module}.{scale}_global_scale"
            if device == "cpu":
                torch.testing.assert_close(result[converted_name], 1 / tensor)
        else:
            converted_name = name

        expected_names.add(converted_name)
        if param not in {"input_scale", "weight_scale_2"}:
            assert result[converted_name] is tensor
    assert set(result) == expected_names


@pytest.mark.unit
@pytest.mark.parametrize(
    ("missing_param", "missing_converted_param"),
    [
        ("weight_scale", "weight_scale"),
        ("weight_scale_2", "weight_global_scale"),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_modelopt_nvfp4_validate_requires_weight_scales(
    modelopt_nvfp4_tensors, missing_param, missing_converted_param, device
):
    module = "model.layers.0.self_attn.q_proj"
    tensors = {
        name: tensor.to(device) for name, tensor in modelopt_nvfp4_tensors.items()
    }
    del tensors[f"{module}.{missing_param}"]

    with pytest.raises(
        ValueError, match=f"without corresponding {module}.{missing_converted_param}"
    ):
        ModelOptNvfp4Converter().validate(tensors)


@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_modelopt_nvfp4_validate_rejects_non_uint8_packed_weight(device):
    module = "model.layers.0.self_attn.q_proj"
    with torch.device(device):
        tensors = {
            f"{module}.weight_packed": torch.ones(64, 32, dtype=torch.bfloat16),
            f"{module}.weight_scale": torch.ones(64, 4).to(torch.float8_e4m3fn),
            f"{module}.weight_global_scale": torch.tensor(4.0),
            f"{module}.input_global_scale": torch.tensor(2.0),
        }

    with pytest.raises(ValueError, match="weight_packed.*torch.uint8"):
        ModelOptNvfp4Converter().validate(tensors)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs", [{"ignore": ["lm_head"]}, {"targets": [r"re:.*proj$"]}]
)
def test_modelopt_nvfp4_validate_leaves_unselected_packed_weights(kwargs):
    tensors = {"lm_head.weight_packed": torch.ones(64, 32, dtype=torch.int32)}

    result = ModelOptNvfp4Converter(**kwargs).validate(dict(tensors))

    assert result == tensors
