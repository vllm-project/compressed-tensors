# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from compressed_tensors.entrypoints.convert import FP8Converter
from safetensors.torch import save_file


def _write_checkpoint(path, config: dict, tensors: dict):
    """Write a local checkpoint with a config file and a safetensors shard."""
    with open(path / "config.json", "w") as f:
        json.dump(config, f)
    save_file(tensors, str(path / "model.safetensors"))


def _fp8_checkpoint_tensors() -> dict:
    return {
        "model.layers.0.mlp.up_proj.weight": torch.zeros(
            256, 256, dtype=torch.float8_e4m3fn
        ),
        "model.layers.0.mlp.up_proj.weight_scale_inv": torch.ones(
            2, 2, dtype=torch.float32
        ),
        "model.layers.0.ffn.experts.0.w1.weight": torch.zeros(
            256, 256, dtype=torch.int8
        ),
        "model.layers.0.ffn.experts.0.w1.scale": torch.ones(8, 8, dtype=torch.float32),
        "model.embed_tokens.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
    }


@pytest.fixture
def fp8_config() -> dict:
    # model_type deliberately not registered in transformers: from_pretrained
    # reads config.json directly and must not depend on AutoConfig
    return {
        "architectures": ["TotallyUnknownForCausalLM"],
        "model_type": "totally_unknown",
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [64, 64],
        },
    }


@pytest.mark.unit
def test_fp8_converter_from_pretrained_reads_config_json(tmp_path, fp8_config):
    _write_checkpoint(tmp_path, fp8_config, _fp8_checkpoint_tensors())

    converter = FP8Converter.from_pretrained(str(tmp_path))

    assert converter.weight_block_size == (64, 64)
    assert converter.ignore == ["lm_head"]
    # both fp8 and fp4 targets are detected from the safetensors headers
    assert any("up_proj" in target for target in converter.fp8_targets)
    assert any("experts" in target for target in converter.fp4_targets)

    config = converter.update_config(None)
    formats = {group.format for group in config.config_groups.values()}
    assert len(formats) == 2  # mixed fp8 + fp4 checkpoint


@pytest.mark.unit
def test_fp8_converter_from_pretrained_cascades_to_text_config(tmp_path, fp8_config):
    # quantization_config nested under text_config, as in composite checkpoints
    config = dict(fp8_config)
    config.pop("quantization_config")
    config["text_config"] = {
        "quantization_config": {
            "quant_method": "fp8",
            "modules_to_not_convert": ["vision_tower"],
        }
    }
    _write_checkpoint(tmp_path, config, _fp8_checkpoint_tensors())

    converter = FP8Converter.from_pretrained(str(tmp_path))

    # default block size when the config does not specify one
    assert converter.weight_block_size == (128, 128)
    assert "re:.*vision_tower.*" in converter.ignore


@pytest.mark.unit
def test_fp8_converter_from_pretrained_requires_quant_config(tmp_path, fp8_config):
    config = dict(fp8_config)
    config.pop("quantization_config")
    _write_checkpoint(tmp_path, config, _fp8_checkpoint_tensors())

    with pytest.raises(ValueError, match="does not contain quantization_config"):
        FP8Converter.from_pretrained(str(tmp_path))


@pytest.mark.unit
def test_fp8_converter_from_pretrained_rejects_non_fp8_config(tmp_path, fp8_config):
    fp8_config["quantization_config"] = {"quant_method": "awq"}
    _write_checkpoint(tmp_path, fp8_config, _fp8_checkpoint_tensors())

    with pytest.raises(ValueError, match="not an fp8 config"):
        FP8Converter.from_pretrained(str(tmp_path))
