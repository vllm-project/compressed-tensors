# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.entrypoints.convert import (
    AutoAWQConverter,
    FP8BlockDequantizer,
    ModelOptNvfp4Converter,
)


def _create_fp8_block_tensors(device="meta"):
    """FP8 block-quantized tensors: weight (fp8) + weight_scale_inv (f32)"""
    with torch.device(device):
        return {
            "model.layer0.mlp.up_proj.weight": torch.empty(
                256, 256, dtype=torch.float8_e4m3fn
            ),
            "model.layer0.mlp.up_proj.weight_scale_inv": torch.empty(
                2, 2, dtype=torch.float32
            ),
            "model.embed_tokens.weight": torch.empty(128, 128, dtype=torch.bfloat16),
        }


@pytest.mark.unit
def test_validate_chain_propagates_meta_tensors():
    """
    Core test for Brian's feedback: converter 2's validate must receive
    converter 1's output, not the pristine input tensors.
    """
    dequantizer = FP8BlockDequantizer(
        targets=[r"re:.*layer\d+\.mlp\..*proj$"],
        weight_block_size=(128, 128),
    )

    tensors = _create_fp8_block_tensors()

    # After dequantizer validate: fp8 weight + scale_inv -> bfloat16 weight
    result = dequantizer.validate(tensors)

    assert "model.layer0.mlp.up_proj.weight" in result
    assert result["model.layer0.mlp.up_proj.weight"].dtype == torch.bfloat16
    assert "model.layer0.mlp.up_proj.weight_scale_inv" not in result
    assert "model.embed_tokens.weight" in result


@pytest.mark.unit
def test_validate_chain_multiple_converters():
    """
    Validate that chaining validate calls through a list of converters
    works the same way the pipeline does it in validate_file.
    """
    dequantizer = FP8BlockDequantizer(
        targets=[r"re:.*layer\d+\.mlp\..*proj$"],
        weight_block_size=(128, 128),
    )

    tensors = _create_fp8_block_tensors()
    converters = [dequantizer]

    # Simulate validate_file loop
    for converter in converters:
        tensors = converter.validate(tensors)

    assert "model.layer0.mlp.up_proj.weight" in tensors
    assert tensors["model.layer0.mlp.up_proj.weight"].dtype == torch.bfloat16
    assert "model.layer0.mlp.up_proj.weight_scale_inv" not in tensors


@pytest.mark.unit
def test_config_chain_dequantizer_then_requantizer():
    """
    Config chain: dequantizer returns None, re-quantizer returns its config.
    The re-quantizer receives None (not a stale config from the dequantizer).
    """
    dequantizer = FP8BlockDequantizer(targets=["Linear"])
    requantizer = ModelOptNvfp4Converter(targets=[r"re:.*proj$"])

    config = None
    for converter in [dequantizer, requantizer]:
        config = converter.update_config(config)

    # Dequantizer returned None, requantizer got None and returned its own config
    assert config is not None
    assert len(config.config_groups) == 1
    assert "config_group_0" in config.config_groups


@pytest.mark.unit
def test_config_chain_two_requantizers_merge():
    """
    Config chain: two re-quantizers merge their configs via
    QuantizationConfig.merge(). Result has config groups from both.
    """
    converter1 = AutoAWQConverter(group_size=128, targets=["Linear"])
    converter2 = ModelOptNvfp4Converter(targets=[r"re:.*proj$"])

    config = None
    for converter in [converter1, converter2]:
        config = converter.update_config(config)

    assert config is not None
    assert len(config.config_groups) == 2


@pytest.mark.unit
def test_single_converter_backward_compat():
    """
    A single converter passed through the list-based pipeline should
    produce the same config as calling update_config(None) directly.
    """
    converter = AutoAWQConverter(group_size=128, targets=["Linear"])

    # Direct call
    direct_config = converter.update_config(None)

    # Pipeline-style (list of one)
    config = None
    for c in [converter]:
        config = c.update_config(config)

    assert config is not None
    assert direct_config is not None
    assert config.config_groups.keys() == direct_config.config_groups.keys()
    assert config.format == direct_config.format
    assert config.quantization_status == direct_config.quantization_status
