# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import runpy
from pathlib import Path

import datasets
import pytest
import torch
from compressed_tensors.quantization import QuantizationConfig
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def local_calibration(monkeypatch):
    """Replace Hub inputs with a small model and two chat samples."""
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        chat_template=(
            "{% for message in messages %}{{ message['content'] }} {% endfor %}"
        ),
    )
    dataset = datasets.Dataset.from_dict(
        {
            "messages": [
                [{"role": "user", "content": "hello world"}],
                [{"role": "user", "content": "world hello"}],
            ]
        }
    )

    def load_model(*args, **kwargs):
        # Both input dimensions must be divisible by the INT4 group size (128).
        return LlamaForCausalLM(
            LlamaConfig(
                vocab_size=16,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=32,
                tie_word_embeddings=False,
            )
        )

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: dataset)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "script, output_dir, config_file, formats, weight_suffix, weight_dtype",
    [
        (
            "compress_int8.py",
            "Meta-Llama-3-8B-Instruct-W8A8",
            "int8_config.json",
            {"int-quantized"},
            ".weight",
            torch.int8,
        ),
        (
            "compress_w4a16.py",
            "Meta-Llama-3-8B-W4A16",
            "int4_config.json",
            {"pack-quantized"},
            ".weight_packed",
            torch.int32,
        ),
        (
            "compress_nvfp4.py",
            "TinyModel-NVFP4",
            None,
            {"nvfp4-pack-quantized"},
            ".weight_packed",
            torch.uint8,
        ),
        (
            "compress_nvfp4_fp8.py",
            "Meta-Llama-3-8B-Instruct-FP8-NVFP4",
            "mixed_precision_config.json",
            {"float-quantized", "nvfp4-pack-quantized"},
            ".weight_packed",
            torch.uint8,
        ),
    ],
)
def test_compression_example(
    script,
    output_dir,
    config_file,
    formats,
    weight_suffix,
    weight_dtype,
    tmp_path,
    monkeypatch,
    local_calibration,
):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")

    output = tmp_path / output_dir
    saved_config = json.loads((output / "config.json").read_text())
    quantization = saved_config["quantization_config"]
    config = QuantizationConfig.model_validate(quantization)
    assert quantization["quant_method"] == "compressed-tensors"
    assert config.quantization_status == "compressed"
    assert {group.format for group in config.config_groups.values()} == formats
    assert config.format == (
        next(iter(formats)) if len(formats) == 1 else "mixed-precision"
    )

    if config_file is not None:
        expected = QuantizationConfig.model_validate_json(
            (EXAMPLES / config_file).read_text()
        )
        assert config.ignore == expected.ignore
        assert len(config.config_groups) == len(expected.config_groups)
        actual_by_targets = {
            tuple(group.targets): group for group in config.config_groups.values()
        }
        for scheme in expected.config_groups.values():
            actual = actual_by_targets[tuple(scheme.targets)]
            assert actual.targets == scheme.targets
            assert actual.weights == scheme.weights
            assert actual.input_activations == scheme.input_activations
    else:
        assert len(config.config_groups) == 1
        scheme = next(iter(config.config_groups.values()))
        assert scheme.weights.num_bits == 4
        assert scheme.weights.group_size == 16
        assert scheme.input_activations.num_bits == 4

    files = list(output.glob("*.safetensors"))
    assert files
    tensors = {
        name: tensor for file in files for name, tensor in load_file(file).items()
    }
    weights = [
        tensor for name, tensor in tensors.items() if name.endswith(weight_suffix)
    ]
    assert weights
    assert any(tensor.dtype == weight_dtype for tensor in weights)
    scales = [
        tensor for name, tensor in tensors.items() if name.endswith("weight_scale")
    ]
    assert scales
    assert all(torch.isfinite(scale.float()).all() for scale in scales)
    assert all((scale.float() > 0).all() for scale in scales)
    if len(formats) > 1:
        attention_weights = [
            tensor
            for name, tensor in tensors.items()
            if ".self_attn." in name and name.endswith(".weight")
        ]
        assert attention_weights
        assert all(tensor.dtype == torch.float8_e4m3fn for tensor in attention_weights)
