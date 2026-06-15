# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.entrypoints.convert import ModelOptNvfp4Converter


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
    Test that the converter's validate method works correctly with meta tensors.
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

    # Should not raise any errors
    converter.validate(tensors)
