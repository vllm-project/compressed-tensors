# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import shutil
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle import apply_quantization_config
from compressed_tensors.utils import is_match, match_named_modules
from tests.testing_utils import requires_accelerate
from transformers import AutoModelForCausalLM


@pytest.fixture(scope="module", autouse=True)
def cleanup_model_cache():
    """Clean up the test model cache directory after all tests complete."""
    yield
    try:
        shutil.rmtree("test-apply-model-cache", ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.named_modules.return_value = [
        ("layer1", MagicMock()),
        ("layer2", MagicMock()),
        ("layer3", MagicMock()),
    ]
    return model


@pytest.fixture
def mock_module():
    return MagicMock()


@pytest.fixture
def llama_stories_model():
    return AutoModelForCausalLM.from_pretrained(
        "Xenova/llama2.c-stories15M",
        torch_dtype="auto",
        cache_dir="test-apply-model-cache",
    )


def test_target_prioritization(mock_frozen):
    # tests that the config_groups are applied in the correct order
    # of priority, where exact layer name > regex > module name
    config = {
        "quant_method": "compressed-tensors",
        "format": "fakequant",
        "config_groups": {
            "group_1": {
                "weights": {
                    "num_bits": 8,
                },
                "targets": ["Linear"],
            },
            "group_2": {
                "weights": {
                    "num_bits": 4,
                },
                "targets": ["re:.*down_proj"],
            },
            "group_3": {
                "weights": {
                    "num_bits": 2,
                },
                "targets": ["model.layers.0.mlp.down_proj"],
            },
        },
    }

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceM4/tiny-random-LlamaForCausalLM",
        torch_dtype="auto",
        cache_dir="test-apply-model-cache",
    )
    model.eval()

    config = QuantizationConfig(**config)
    config.quantization_status = QuantizationStatus.CALIBRATION
    apply_quantization_config(model, config)
    mock_frozen(model)

    for name, module in model.named_modules():
        if name == "model.layers.0.mlp.down_proj":
            assert module.quantization_scheme.weights.num_bits == 2
        elif re.match(".*down_proj", name):
            assert module.quantization_scheme.weights.num_bits == 4
        elif isinstance(module, torch.nn.Linear):
            assert module.quantization_scheme.weights.num_bits == 8


def test_apply_quantization_config_tinyllama():
    quant_config = get_sample_tinyllama_quant_config(
        status=QuantizationStatus.INITIALIZED
    )
    model = get_tinyllama_model()

    # check that model is not already quantized
    for module in model.modules():
        _test_layer_quantization_status(module, inputs=False, weights=False)

    # apply quant config to model
    apply_quantization_config(model, quant_config)

    # check for correct application of quant config
    for quant_scheme in quant_config.config_groups.values():
        for name, module in match_named_modules(
            model, quant_scheme.targets, quant_config.ignore
        ):
            _test_layer_quantization_status(
                module,
                inputs=quant_scheme.input_activations is not None,
                weights=quant_scheme.weights is not None,
                expected_status=QuantizationStatus.INITIALIZED,
            )


@pytest.mark.parametrize(
    "config",
    [
        QuantizationConfig(
            config_groups={
                "linear": QuantizationScheme(
                    targets=["Linear"],
                    input_activations=QuantizationArgs(
                        num_bits=8, type="float", strategy="tensor"
                    ),
                )
            }
        ),
        QuantizationConfig(
            config_groups={
                "linear": QuantizationScheme(
                    targets=["Linear"],
                    input_activations=QuantizationArgs(
                        num_bits=8, type="float", strategy="tensor"
                    ),
                )
            },
            ignore=[
                "model.layers.0.self_attn.q_proj",
                "model.layers.1.self_attn.k_proj",
                "model.layers.2.self_attn.v_proj",
            ],
        ),
        QuantizationConfig(
            config_groups={},
            kv_cache_scheme=QuantizationArgs(
                num_bits=8, type="float", strategy="tensor"
            ),
        ),
        QuantizationConfig(
            config_groups={
                "attention": QuantizationScheme(
                    targets=["LlamaAttention"],
                    input_activations=QuantizationArgs(
                        num_bits=8, type="float", strategy="tensor"
                    ),
                )
            },
            kv_cache_scheme=QuantizationArgs(
                num_bits=8, type="float", strategy="tensor"
            ),
        ),
    ],
)
def test_from_pretrained(config: QuantizationConfig):
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")
    apply_quantization_config(model, config)
    _config = QuantizationConfig.from_pretrained(model)
    assert list(_config.config_groups.values()) == list(config.config_groups.values())
    assert _config.kv_cache_scheme == config.kv_cache_scheme
    assert _config.ignore == config.ignore


def test_serialize_config_tinyllama():
    quant_config = get_sample_tinyllama_quant_config()
    model = get_tinyllama_model()

    # check that model is not already quantized
    for module in model.modules():
        _test_layer_quantization_status(module, inputs=False, weights=False)

    # apply quant config to model
    apply_quantization_config(model, quant_config)

    serialized_config = QuantizationConfig.from_pretrained(model)
    assert len(serialized_config.config_groups) == 2
    assert serialized_config.config_groups["group_0"].targets == ["Embedding"]
    assert serialized_config.config_groups["group_0"].input_activations is None
    assert serialized_config.config_groups["group_1"].targets == ["Linear"]
    assert serialized_config.config_groups["group_1"].input_activations is not None
    assert serialized_config.format == CompressionFormat.dense.value
    assert serialized_config.quant_method == DEFAULT_QUANTIZATION_METHOD
    assert serialized_config.ignore == ["model.layers.1.mlp.down_proj"]
    if serialized_config.global_compression_ratio is not None:
        assert serialized_config.global_compression_ratio > 1.0
        assert serialized_config.global_compression_ratio < 8.0


def _test_layer_quantization_status(
    module,
    inputs: bool,
    weights: bool,
    expected_status: Optional[QuantizationStatus] = None,
    expected_dtype: Optional[torch.dtype] = None,
):
    # check if quantization is applied at all (true if inputs or weights targeted)
    quantized = inputs or weights
    assert hasattr(module, "quantization_scheme") == quantized
    assert hasattr(module, "quantization_status") == quantized
    if expected_status is not None:
        assert module.quantization_status is expected_status

    # check inputs matches expected
    assert hasattr(module, "input_scale") == inputs
    assert hasattr(module, "input_zero_point") == inputs

    # check weights matches expected
    assert hasattr(module, "weight_scale") == weights
    assert hasattr(module, "weight_zero_point") == weights
    if weights and expected_dtype is not None:
        assert module.weight.dtype is expected_dtype


def get_tinyllama_model():
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        torch_dtype="auto",
        cache_dir="test-apply-model-cache",
    )


def get_sample_tinyllama_quant_config(
    status: QuantizationStatus = QuantizationStatus.FROZEN,
):
    config_dict = {
        "quant_method": "compressed-tensors",
        "format": "fakequant",
        "quantization_status": status,
        "global_compression_ratio": None,
        "config_groups": {
            "group_1": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                },
                "targets": ["Linear"],
            },
            "group_2": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "tensor",
                },
                "input_activations": None,
                "targets": ["Embedding"],
            },
        },
        "ignore": ["LlamaRotaryEmbedding", "model.layers.1.mlp.down_proj"],
    }
    return QuantizationConfig.model_validate(config_dict)


@requires_accelerate()
@pytest.mark.parametrize(
    "target,should_raise_warning",
    [
        [("Linear",), False],
        [("Linear", "re:.*foobarbaz"), True],
    ],
)
def test_apply_quantization_config(caplog, target, should_raise_warning):
    import logging

    # load a dense, unquantized tiny llama model
    model = get_tinyllama_model()
    quantization_config_dict = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "config_groups": {
            "group_1": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "tensor",
                },
                "targets": target,
            }
        },
        "ignore": ["lm_head", "re:.*gate"],
    }

    config = QuantizationConfig(**quantization_config_dict)
    config.quantization_status = QuantizationStatus.CALIBRATION

    # mismatch in the ignore key of quantization_config_dict
    with caplog.at_level(logging.WARNING):
        apply_quantization_config(model, config)
        if should_raise_warning:
            assert len(caplog.text) > 0
        else:
            assert len(caplog.text) == 0


def test_multi_apply_quantization_config():
    """
    Ensure that multiple quantization configs are applied correctly
    If quantization config was previously applied to a module,
    those changes should be reset for newly applied quantization config
    """
    model = get_tinyllama_model()

    # FP8 applied to self_attn
    qconfig1 = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\.(k|q|o|v)_proj$",
                ],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
            )
        },
        ignore=["lm_head"],
    )
    # W4A16_ASYM applied to mlp and self_attn.o_proj to validate overwriting
    qconfig2 = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=[
                    r"re:.*mlp\.(down|gate|up)_proj$",
                    r"re:.*self_attn\.o_proj$",
                ],
                weights=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                    symmetric=False,
                    dynamic=False,
                ),
            )
        },
        ignore=["lm_head"],
    )

    apply_quantization_config(model, qconfig1)
    apply_quantization_config(model, qconfig2)
    for name, module in model.named_modules():
        if is_match(
            name, module, qconfig2.config_groups["group_0"].targets, qconfig2.ignore
        ):
            # assert W4A16_ASYM parameters are present with correct shape
            # and FP8 parameters have been removed
            assert not hasattr(module, "input_scale")
            assert not hasattr(module, "input_zero_point")
            weight_scale = getattr(module, "weight_scale", None)
            assert (
                weight_scale is not None
                and weight_scale.shape[:-1] == module.weight.shape[:-1]
                and weight_scale.shape[-1] == module.weight.shape[-1] / 128
            )
            weight_zero_point = getattr(module, "weight_zero_point", None)
            assert (
                weight_zero_point is not None
                and weight_zero_point.shape[:-1] == module.weight.shape[:-1]
                and weight_zero_point.shape[-1] == module.weight.shape[-1] / 128
            )

        elif is_match(
            name, module, qconfig1.config_groups["group_0"].targets, qconfig1.ignore
        ):
            # assert FP8 scheme parameters are present with correct shape
            input_scale = getattr(module, "input_scale", None)
            assert input_scale is not None and input_scale.shape == torch.Size([1])
            input_zero_point = getattr(module, "input_zero_point", None)
            assert (
                input_zero_point is not None
                and input_zero_point.shape == torch.Size([1])
            )
            weight_scale = getattr(module, "weight_scale", None)
            assert weight_scale is not None and weight_scale.shape == torch.Size([1])
            weight_zero_point = getattr(module, "weight_zero_point", None)
            assert (
                weight_zero_point is not None
                and weight_zero_point.shape == torch.Size([1])
            )


@requires_accelerate()
def test_apply_kv_cache():
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    args = QuantizationArgs(num_bits=8, type="float", strategy="tensor")
    config = QuantizationConfig(config_groups={}, kv_cache_scheme=args)

    apply_quantization_config(model, config)

    for layer in model.model.layers:
        assert getattr(layer.self_attn, "quantization_scheme").input_activations == args
        assert hasattr(layer.self_attn, "k_scale")
        assert hasattr(layer.self_attn, "v_scale")


@requires_accelerate()
def test_apply_attention():
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    scheme = QuantizationScheme(
        targets=["LlamaAttention"],
        input_activations=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
    )
    config = QuantizationConfig(config_groups={"attention": scheme})

    apply_quantization_config(model, config)

    for layer in model.model.layers:
        assert getattr(layer.self_attn, "quantization_scheme") == scheme
        assert hasattr(layer.self_attn, "q_scale")
        assert hasattr(layer.self_attn, "k_scale")
        assert hasattr(layer.self_attn, "v_scale")


def test_group_size_validation_raises_error():
    """Test that GROUP strategy validation raises error for non-divisible layers"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    # Create config with GROUP strategy and group_size=128
    # Most layers have 288 input features (NOT divisible by 128)
    # down_proj layers have 768 input features (divisible by 128)
    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                ),
            )
        },
    )

    # Should raise ValueError because most layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj)
    # have 288 input features which are not divisible by group_size=128
    with pytest.raises(ValueError, match="Quantization divisibility validation failed"):
        apply_quantization_config(model, config, validate_group_or_block_size=True)


def test_group_size_validation_error_message():
    """Test that validation error message contains helpful information"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                ),
            )
        },
    )

    try:
        apply_quantization_config(model, config, validate_group_or_block_size=True)
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        # Check that error message contains expected components
        assert "Quantization divisibility validation failed" in error_msg
        # Should contain some layer names with 288 input features
        assert "q_proj" in error_msg or "k_proj" in error_msg or "gate_proj" in error_msg
        assert "ignore:" in error_msg
        assert "SUGGESTED FIX" in error_msg


def test_group_size_validation_disabled():
    """Test that validation can be disabled"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                ),
            )
        },
    )

    # Should NOT raise error when validation is disabled
    apply_quantization_config(model, config, validate_group_or_block_size=False)

    # Verify that quantization was still applied
    assert hasattr(model.model.layers[0].self_attn.q_proj, "quantization_scheme")
    assert hasattr(model.model.layers[0].mlp.down_proj, "quantization_scheme")


def test_group_size_validation_with_ignore_list():
    """Test that validation respects ignore list"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    # Create config with problematic layers (288 input features) in ignore list
    # Only quantize down_proj layers which have 768 input features (divisible by 128)
    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                ),
            )
        },
        ignore=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj",
                "re:.*gate_proj", "re:.*up_proj", "lm_head"],
    )

    # Should NOT raise error because problematic layers are ignored
    apply_quantization_config(model, config, validate_group_or_block_size=True)

    # Verify that only down_proj layers were quantized
    assert hasattr(model.model.layers[0].mlp.down_proj, "quantization_scheme")
    assert not hasattr(model.model.layers[0].self_attn.q_proj, "quantization_scheme")
    assert not hasattr(model.model.layers[0].mlp.gate_proj, "quantization_scheme")


def test_channel_strategy_no_validation():
    """Test that validation doesn't trigger for non-GROUP strategies"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    # Create config with CHANNEL strategy (not GROUP)
    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="channel",
                ),
            )
        },
    )

    # Should NOT raise error for CHANNEL strategy
    apply_quantization_config(model, config, validate_group_or_block_size=True)

    # Verify that quantization was applied
    assert hasattr(model.model.layers[0].self_attn.q_proj, "quantization_scheme")
    assert hasattr(model.model.layers[0].mlp.down_proj, "quantization_scheme")


def test_tensor_strategy_no_validation():
    """Test that validation doesn't trigger for TENSOR strategy"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    # Create config with TENSOR strategy (not GROUP)
    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type="int",
                    symmetric=True,
                    strategy="tensor",
                ),
            )
        },
    )

    # Should NOT raise error for TENSOR strategy
    apply_quantization_config(model, config, validate_group_or_block_size=True)

    # Verify that quantization was applied
    assert hasattr(model.model.layers[0].self_attn.q_proj, "quantization_scheme")
    assert hasattr(model.model.layers[0].mlp.down_proj, "quantization_scheme")


def test_block_strategy_validation_raises_error():
    """Test that BLOCK strategy validation raises error for non-divisible layers"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    # Create config with BLOCK strategy
    # Using block_structure [100, 100] which won't divide layers with 288 input features
    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="block",
                    block_structure=[100, 100],
                ),
            )
        },
    )

    # Should raise ValueError because layers with 288 input features
    # are not divisible by block_structure [100, 100]
    with pytest.raises(ValueError, match="Quantization divisibility validation failed"):
        apply_quantization_config(model, config, validate_group_or_block_size=True)


def test_block_strategy_validation_passes():
    """Test that BLOCK strategy validation passes when dimensions are divisible"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    # Create config with BLOCK strategy
    # Using block_structure [96, 96] which divides down_proj layers (768x288)
    # 768 % 96 = 0, 288 % 96 = 0
    # but testing with ignore list for others
    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="block",
                    block_structure=[96, 96],
                ),
            )
        },
        ignore=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj",
                "re:.*gate_proj", "re:.*up_proj", "lm_head"],
    )

    # Should NOT raise error
    apply_quantization_config(model, config, validate_group_or_block_size=True)

    # Verify that only down_proj layers were quantized
    assert hasattr(model.model.layers[0].mlp.down_proj, "quantization_scheme")
    assert not hasattr(model.model.layers[0].self_attn.q_proj, "quantization_scheme")


def test_group_size_validation_with_divisible_group_size():
    """Test that validation passes when all layers are divisible by group_size"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    # Using group_size=96 which divides both 288 and 768
    # 288 % 96 = 0, 768 % 96 = 0
    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="group",
                    group_size=96,
                ),
            )
        },
        ignore=["lm_head"],
    )

    # Should NOT raise error
    apply_quantization_config(model, config, validate_group_or_block_size=True)

    # Verify quantization was applied to layers
    assert hasattr(model.model.layers[0].self_attn.q_proj, "quantization_scheme")
    assert model.model.layers[0].self_attn.q_proj.quantization_scheme.weights.group_size == 96
    assert hasattr(model.model.layers[0].mlp.down_proj, "quantization_scheme")
    assert model.model.layers[0].mlp.down_proj.quantization_scheme.weights.group_size == 96


def test_group_size_validation_with_partial_ignore():
    """Test validation with partial ignore list"""
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    # Ignore only some layers, so other layers with 288 input features should still cause an error
    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                ),
            )
        },
        ignore=["re:.*gate_proj", "lm_head"],  # Only ignore gate_proj, not other 288-input layers
    )

    # Should raise ValueError because layers like q_proj, k_proj, etc. are not divisible and not ignored
    with pytest.raises(ValueError) as exc_info:
        apply_quantization_config(model, config, validate_group_or_block_size=True)

    # Check that error mentions layers that aren't ignored
    error_msg = str(exc_info.value)
    # Should mention some of the non-ignored 288-input layers
    assert any(name in error_msg for name in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj"])
    # Should NOT mention gate_proj since it's ignored
    assert "gate_proj" not in error_msg
