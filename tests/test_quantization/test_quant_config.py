# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_FORMAT,
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.quantization.quant_config import (
    _map_to_checkpoint_names,
    get_vllm_module_type,
)
from pydantic import ValidationError
from transformers import AutoModelForImageTextToText


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


@pytest.mark.parametrize(
    "model_id,hf_ignores,checkpoint_ignores",
    [
        pytest.param(
            "llava-hf/llava-interleave-qwen-0.5b-hf",
            [
                "lm_head",
                "model.vision_tower.encoder.layers.0.self_attn.q_proj",
                "model.multi_modal_projector.linear_1",
            ],
            [
                "lm_head",
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
                "multi_modal_projector.linear_1",
            ],
            id="llava",
        ),
        pytest.param(
            "google/gemma-4-12b-it",
            [
                "lm_head",
                "model.embed_vision.patch_dense",
                "model.embed_vision.multimodal_embedder.embedding_projection",
            ],
            [
                "lm_head",
                "model.vision_embedder.patch_dense",
                "model.embed_vision.embedding_projection",
            ],
            id="gemma4",
        ),
        pytest.param(
            "Qwen/Qwen2-VL-2B-Instruct",
            [
                "lm_head",
                "model.visual.merger.mlp.0",
            ],
            [
                "lm_head",
                "visual.merger.mlp.0",
            ],
            id="qwen2_vl",
        ),
    ],
)
def test_map_to_checkpoint_names(model_id, hf_ignores, checkpoint_ignores):
    """Load a real model and verify that HF module names in the ignore list
    are reverse-mapped to checkpoint key names.

    Uses ``device_map="meta"`` so no weights are materialised -- only the
    model structure and its ``_weight_conversions`` are needed.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.float16, device_map="meta"
    )

    result = _map_to_checkpoint_names(model, hf_ignores)

    assert result == checkpoint_ignores


def test_get_vllm_module_type():
    assert get_vllm_module_type("ExpertMLP") == "ExpertMLP"
    assert get_vllm_module_type("ExpertMLPWithGate") == "ExpertMLPWithGate"
    assert get_vllm_module_type("ExpertMLPWithoutGate") == "ExpertMLPWithoutGate"
    assert get_vllm_module_type("Linear") == "Linear"
    assert get_vllm_module_type("DeepseekV4TopKRouter") == "Linear"
    assert get_vllm_module_type("DeepseekV4HashRouter") == "Linear"
    assert get_vllm_module_type("JetMoeTopKGating") == "Linear"
    assert get_vllm_module_type("Qwen3NextGatedDeltaNet") == "Linear"
    assert get_vllm_module_type("JetMoeTopKGating") == "Linear"
