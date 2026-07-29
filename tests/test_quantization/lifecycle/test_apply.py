# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import shutil
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_METHOD,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle import apply_quantization_config
from compressed_tensors.utils import is_match, match_named_modules
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
                        num_bits=8,
                        type="float",
                        strategy="tensor",
                        scale_dtype=FP8_E4M3_DATA.dtype,
                        zp_dtype=torch.float,
                    ),
                )
            }
        ),
        QuantizationConfig(
            config_groups={
                "linear": QuantizationScheme(
                    targets=["Linear"],
                    input_activations=QuantizationArgs(
                        num_bits=8,
                        type="float",
                        strategy="tensor",
                        scale_dtype=FP8_E4M3_DATA.dtype,
                        zp_dtype=torch.float,
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
                num_bits=8,
                type="float",
                strategy="tensor",
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=torch.float,
            ),
        ),
        QuantizationConfig(
            config_groups={
                "attention": QuantizationScheme(
                    targets=["LlamaAttention"],
                    input_activations=QuantizationArgs(
                        num_bits=8,
                        type="float",
                        strategy="tensor",
                        scale_dtype=FP8_E4M3_DATA.dtype,
                        zp_dtype=torch.float,
                    ),
                )
            },
            kv_cache_scheme=QuantizationArgs(
                num_bits=8,
                type="float",
                strategy="tensor",
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=torch.float,
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


def test_apply_kv_cache():
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=torch.float,
    )
    config = QuantizationConfig(config_groups={}, kv_cache_scheme=args)

    apply_quantization_config(model, config)

    for layer in model.model.layers:
        assert getattr(layer.self_attn, "quantization_scheme").input_activations == args
        assert hasattr(layer.self_attn, "k_scale")
        assert hasattr(layer.self_attn, "v_scale")


def test_apply_kv_cache_skips_non_cache_attention():
    class TextAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = torch.nn.Linear(4, 4)

        def forward(self, hidden_states, past_key_value=None):
            return hidden_states

    class VisionAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = torch.nn.Linear(4, 4)

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class CompositeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = CompositeConfig()
            self.text_attention = TextAttention()
            self.vision_attention = VisionAttention()

    class CompositeConfig:
        def __init__(self):
            self.text_config = SimpleNamespace(
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
            )
            self.vision_config = SimpleNamespace(model_type="vision")
            self.decoder = None

        def get_text_config(self, decoder=False):
            self.decoder = decoder
            return self.text_config

    model = CompositeModel()
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=torch.float,
    )
    config = QuantizationConfig(config_groups={}, kv_cache_scheme=args)

    apply_quantization_config(model, config)

    assert model.config.decoder is True
    assert model.config.vision_config.model_type == "vision"
    assert hasattr(model.text_attention, "kv_cache")
    assert model.text_attention.kv_cache.config is model.config.text_config
    assert hasattr(model.text_attention, "k_scale")
    assert hasattr(model.text_attention, "v_scale")
    assert not hasattr(model.vision_attention, "quantization_scheme")
    assert not hasattr(model.vision_attention, "kv_cache")
    assert not hasattr(model.vision_attention, "k_scale")
    assert not hasattr(model.vision_attention, "v_scale")


def test_apply_attention():
    model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    scheme = QuantizationScheme(
        targets=["LlamaAttention"],
        input_activations=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            scale_dtype=FP8_E4M3_DATA.dtype,
            zp_dtype=torch.float,
        ),
    )
    config = QuantizationConfig(config_groups={"attention": scheme})

    apply_quantization_config(model, config)

    for layer in model.model.layers:
        assert getattr(layer.self_attn, "quantization_scheme") == scheme
        assert hasattr(layer.self_attn, "q_scale")
        assert hasattr(layer.self_attn, "k_scale")
        assert hasattr(layer.self_attn, "v_scale")


@pytest.mark.parametrize(
    "config",
    [
        QuantizationConfig(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=8, type="int", symmetric=True, strategy="tensor"
                    ),
                )
            },
            ignore=["lm_head"],
        ),
        QuantizationConfig(
            config_groups={},
            kv_cache_scheme=QuantizationArgs(
                num_bits=8,
                type="float",
                strategy="tensor",
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=torch.float,
            ),
        ),
        QuantizationConfig(
            config_groups={
                "attention": QuantizationScheme(
                    targets=["LlamaAttention"],
                    input_activations=QuantizationArgs(
                        num_bits=8,
                        type="float",
                        strategy="tensor",
                        scale_dtype=FP8_E4M3_DATA.dtype,
                        zp_dtype=torch.float,
                    ),
                )
            },
        ),
        QuantizationConfig(
            config_groups={
                "attention": QuantizationScheme(
                    targets=["LlamaAttention"],
                    input_activations=QuantizationArgs(
                        num_bits=8,
                        type="float",
                        strategy="tensor",
                        scale_dtype=FP8_E4M3_DATA.dtype,
                        zp_dtype=torch.float,
                    ),
                )
            },
            kv_cache_scheme=QuantizationArgs(
                num_bits=8,
                type="float",
                strategy="tensor",
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=torch.float,
            ),
        ),
    ],
    ids=["w8_linear", "fp8_kv_cache", "fp8_attention", "fp8_attention_and_kv_cache"],
)
def test_allowed_modules_per_layer_equivalence(config):
    """
    Apply a quantization config to each transformer layer of a small Llama model
    individually (using allowed_modules) and verify the result is identical to a
    single full apply_quantization_config call.
    """
    model_id = "inference-optimization/Llama-3.2-0.5B-Instruct"

    # Reference: apply all at once
    model_full = AutoModelForCausalLM.from_pretrained(model_id)
    apply_quantization_config(model_full, config, show_progress=False)

    # Per-layer: apply one transformer layer at a time using absolute module names
    model_layerwise = AutoModelForCausalLM.from_pretrained(model_id)
    num_layers = len(model_layerwise.model.layers)
    for i in range(num_layers):
        layer_prefix = f"model.layers.{i}"
        layer = model_layerwise.get_submodule(layer_prefix)
        allowed = {
            f"{layer_prefix}.{name}" if name else layer_prefix
            for name, _ in layer.named_modules()
        }
        apply_quantization_config(
            model_layerwise, config, allowed_modules=allowed, show_progress=False
        )
    # Apply everything outside the transformer layers (e.g. embed_tokens, lm_head)
    top_level = {
        name
        for name, _ in model_layerwise.named_modules()
        if not name.startswith("model.layers")
    }
    apply_quantization_config(
        model_layerwise, config, allowed_modules=top_level, show_progress=False
    )

    # State dicts must have the same keys and tensor shapes.
    # Scale/zero_point buffers are initialized with torch.empty so their values
    # are uninitialized; we compare values only for original model weights.
    sd_full = model_full.state_dict()
    sd_layerwise = model_layerwise.state_dict()
    assert sd_full.keys() == sd_layerwise.keys(), "State dict keys differ"
    qparam_suffixes = ("_scale", "_zero_point", "_g_idx")
    for key in sd_full:
        a, b = sd_full[key], sd_layerwise[key]
        assert a.shape == b.shape and a.dtype == b.dtype, f"Shape/dtype mismatch: {key}"
        if not key.endswith(qparam_suffixes):
            assert torch.equal(a, b), f"Value mismatch in model weight: {key}"

    # Serialized configs must be identical
    config_full = QuantizationConfig.from_pretrained(model_full)
    config_layerwise = QuantizationConfig.from_pretrained(model_layerwise)
    assert list(config_full.config_groups.values()) == list(
        config_layerwise.config_groups.values()
    )
    assert config_full.kv_cache_scheme == config_layerwise.kv_cache_scheme
    assert config_full.ignore == config_layerwise.ignore


linear_scheme = QuantizationScheme(targets=["Linear"])
attention_scheme = QuantizationScheme(
    targets=["LlamaAttention"],
    input_activations=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
)
attention_linears = QuantizationScheme(targets=[r"re:.*self_attn\..*"])
mlp_linears = QuantizationScheme(targets=[r"re:.*mlp\..*"])
down_proj_scheme = QuantizationScheme(targets=["re:.*down_proj"])


@pytest.mark.parametrize(
    "config, expected_schemes",
    [
        # all linears
        (
            QuantizationConfig(config_groups={"group_0": linear_scheme}),
            {
                p: linear_scheme
                for p in [
                    f"model.layers.{i}.self_attn.{k}_proj"
                    for i in range(6)
                    for k in "qkvo"
                ]
                + [
                    f"model.layers.{i}.mlp.{k}_proj"
                    for i in range(6)
                    for k in ["gate", "up", "down"]
                ]
                + ["lm_head"]
            },
        ),
        # only attention
        (
            QuantizationConfig(config_groups={"group_0": attention_scheme}),
            {f"model.layers.{i}.self_attn": attention_scheme for i in range(6)},
        ),
        # linear and attention
        (
            QuantizationConfig(
                config_groups={"attention": attention_scheme, "linear": linear_scheme},
            ),
            {
                **{f"model.layers.{i}.self_attn": attention_scheme for i in range(6)},
                **{
                    p: linear_scheme
                    for p in [
                        f"model.layers.{i}.self_attn.{k}_proj"
                        for i in range(6)
                        for k in "qkvo"
                    ]
                    + [
                        f"model.layers.{i}.mlp.{k}_proj"
                        for i in range(6)
                        for k in ["gate", "up", "down"]
                    ]
                    + ["lm_head"]
                },
            },
        ),
        # only down proj
        (
            QuantizationConfig(config_groups={"group_0": down_proj_scheme}),
            {f"model.layers.{i}.mlp.down_proj": down_proj_scheme for i in range(6)},
        ),
        # attention linears and mlp linears as separate groups
        (
            QuantizationConfig(
                config_groups={
                    "attention_linears": attention_linears,
                    "mlp_linears": mlp_linears,
                },
            ),
            {
                **{
                    f"model.layers.{i}.self_attn.{k}_proj": attention_linears
                    for i in range(6)
                    for k in "qkvo"
                },
                **{
                    f"model.layers.{i}.mlp.{k}_proj": mlp_linears
                    for i in range(6)
                    for k in ["gate", "up", "down"]
                },
            },
        ),
    ],
)
def test_apply_model(config, expected_schemes):
    model = AutoModelForCausalLM.from_pretrained(
        "nm-testing/tinysmokellama-3.2",
        cache_dir="test-apply-model-cache",
    )
    apply_quantization_config(model, config)

    for name, module in model.named_modules():
        if name in expected_schemes:
            assert hasattr(
                module, "quantization_scheme"
            ), f"{name} should have quantization_scheme"
            assert (
                module.quantization_scheme == expected_schemes[name]
            ), f"{name} has wrong scheme"
        else:
            assert not hasattr(
                module, "quantization_scheme"
            ), f"{name} should not have quantization_scheme"
