# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import pytest

from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.quantization.quant_scheme import NVFP4
from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    ModelOptNvfp4Converter,
)
from compressed_tensors.quantization import QuantizationArgs, QuantizationType


def test_convert_checkpoint(tmp_path):
    MODEL_ID = "nvidia/Qwen3-8B-NVFP4"
    convert_outdir = tmp_path / "convert_out"

    right_targets = [
        "re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
        "re:.*self_attn.*\.(q|k|v|o)_proj$",
    ]
    wrong_targets = [
        "re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
        "re:.*self_attn.*\.(q|k|o)_proj$",
    ]
    right_kv_cache_scheme = QuantizationArgs(
        num_bits=8, dynamic=False, type=QuantizationType.FLOAT
    )
    wrong_kv_cache_scheme = None

    with pytest.raises(ValueError):
        convert_checkpoint(
            model_stub=MODEL_ID,
            save_directory=convert_outdir,
            converter=ModelOptNvfp4Converter(
                targets=right_targets,
                kv_cache_scheme=wrong_kv_cache_scheme,
            ),
        )

    with pytest.raises(ValueError):
        convert_checkpoint(
            model_stub=MODEL_ID,
            save_directory=convert_outdir,
            converter=ModelOptNvfp4Converter(
                targets=wrong_targets,
                kv_cache_scheme=right_kv_cache_scheme,
            ),
        )

    convert_checkpoint(
        model_stub=MODEL_ID,
        save_directory=convert_outdir,
        converter=ModelOptNvfp4Converter(
            targets=right_targets,
            kv_cache_scheme=right_kv_cache_scheme,
        ),
    )

    with open(convert_outdir / "config.json", "r") as f:
        config = json.load(f)

        qconfig = QuantizationConfig.model_validate(config["quantization_config"])

    assert qconfig.format == "nvfp4-pack-quantized"
    assert qconfig.quant_method == "compressed-tensors"
    assert len(qconfig.config_groups) == 1
    # assert weights and input_activations are a superset of what's in the NVFP4 preset
    assert (
        qconfig.config_groups["config_group_0"].weights.model_dump().items()
        >= NVFP4["weights"].model_dump().items()
    )
    assert (
        qconfig.config_groups["config_group_0"].input_activations.model_dump().items()
        >= NVFP4["input_activations"].model_dump().items()
    )
