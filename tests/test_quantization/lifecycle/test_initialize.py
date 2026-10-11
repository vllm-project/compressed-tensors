# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace
from unittest.mock import Mock

import compressed_tensors.quantization.lifecycle.initialize as initialize_lifecycle
import numpy
import pytest
import torch
from compressed_tensors.modeling import IMPL_ATTR
from compressed_tensors.offload import set_onload_device
from compressed_tensors.quantization import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.initialize import (
    _is_attention_module,
    initialize_attn_qparams,
    initialize_module_for_quantization,
    is_attention_module,
    is_cached_attention_module,
)
from tests.testing_utils import requires_gpu
from torch.nn import Linear
from transformers import PretrainedConfig


NUM_BITS = 8
Q_PARAM_NAMES = {
    "input_activations": "input",
    "weights": "weight",
    "output_activations": "output",
}


class CacheAwareAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, past_key_value=None):
        return hidden_states


class PluralCacheAwareAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, past_key_values=None):
        return hidden_states


class EncoderAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class T5StyleAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, past_key_value=None):
        return hidden_states


class ProjectionOnlyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states):
        return hidden_states


class ProjectionAttention(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states


@pytest.fixture
def layer():
    return Linear(4, 4)


@pytest.mark.parametrize(
    "attention_cls, expected",
    [
        (CacheAwareAttention, True),
        (PluralCacheAwareAttention, True),
        (EncoderAttention, False),
        (T5StyleAttention, False),
        (ProjectionOnlyBlock, False),
    ],
)
def test_is_cached_attention_module(attention_cls, expected):
    assert is_cached_attention_module(attention_cls()) is expected


@pytest.mark.parametrize(
    ("attention_cls", "expected"),
    [
        (CacheAwareAttention, True),
        (T5StyleAttention, True),
        (ProjectionOnlyBlock, False),
    ],
)
def test_is_attention_module_detects_attention_blocks(attention_cls, expected):
    assert _is_attention_module(attention_cls()) is expected


def test_is_attention_module_is_deprecated():
    with pytest.warns(DeprecationWarning, match="is_cached_attention_module"):
        assert is_attention_module(CacheAwareAttention()) is True


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_is_cached_attention_module_warns_and_returns_false_when_uninspectable(
    monkeypatch, error_type
):
    warning = Mock()
    monkeypatch.setattr(
        initialize_lifecycle.inspect,
        "signature",
        Mock(side_effect=error_type("signature unavailable")),
    )
    monkeypatch.setattr(initialize_lifecycle.logger, "warning", warning)

    assert is_cached_attention_module(CacheAwareAttention()) is False

    warning.assert_called_once_with(
        "Unable to inspect an attention module's forward signature; "
        "skipping KV cache quantization for uninspectable modules",
        log_once=True,
    )


class MockKVCache(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def test_initialize_attn_qparams_uses_kv_cache_config():
    config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
    )

    attention = _initialize_attn_qparams(config)

    assert getattr(attention, "kv_cache").config is config
    assert attention.k_scale.shape == (2, 1, 1)
    assert attention.v_scale.shape == (2, 1, 1)


def _initialize_attn_qparams(config):
    attention = CacheAwareAttention()
    attention.kv_cache = MockKVCache(config)
    scheme = QuantizationScheme(
        targets=["CacheAwareAttention"],
        input_activations=QuantizationArgs(strategy=QuantizationStrategy.ATTN_HEAD),
    )

    initialize_attn_qparams(attention, scheme, force_zero_point=False)
    return attention


class LayerAttention(torch.nn.Module):
    def __init__(self, layer_idx, num_kv_heads, head_dim):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.k_proj = torch.nn.Linear(64, num_kv_heads * head_dim)

    def forward(self, hidden_states, past_key_values=None):
        return hidden_states


def _heterogeneous_config():
    if not hasattr(PretrainedConfig, "per_layer_config"):
        pytest.skip("heterogeneous configs require transformers>=5.14")

    config = PretrainedConfig(
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        num_hidden_layers=2,
    )
    config.per_layer_config = {
        1: {"head_dim": 32, "num_key_value_heads": 2, "num_attention_heads": 4}
    }
    return config


def _attention_scheme(**kwargs):
    return QuantizationScheme(
        targets=["LayerAttention"], input_activations=QuantizationArgs(**kwargs)
    )


@pytest.mark.parametrize(
    "strategy,expected_shapes",
    [
        (QuantizationStrategy.TENSOR, [(1,), (1,)]),
        (QuantizationStrategy.ATTN_HEAD, [(4, 1, 1), (2, 1, 1)]),
    ],
)
def test_initialize_attn_qparams_heterogeneous_config(strategy, expected_shapes):
    config = _heterogeneous_config()
    scheme = _attention_scheme(strategy=strategy)

    for layer_idx, expected_shape in enumerate(expected_shapes):
        layer_config = config.per_layer_config[layer_idx]
        attention = LayerAttention(
            layer_idx, layer_config.num_key_value_heads, layer_config.head_dim
        )
        attention.kv_cache = MockKVCache(config)

        initialize_attn_qparams(attention, scheme, force_zero_point=False)

        assert attention.k_scale.shape == expected_shape
        assert attention.v_scale.shape == expected_shape


@pytest.mark.parametrize("layer_idx", [None, 2, -1, True, "1", 1.0])
def test_initialize_attn_qparams_heterogeneous_config_needs_layer_idx(layer_idx):
    attention = LayerAttention(layer_idx, num_kv_heads=4, head_dim=16)
    attention.kv_cache = MockKVCache(_heterogeneous_config())
    scheme = _attention_scheme(strategy=QuantizationStrategy.TENSOR)

    with pytest.raises(ValueError, match="layer_idx"):
        initialize_attn_qparams(attention, scheme, force_zero_point=False)


def test_initialize_attn_qparams_heterogeneous_config_numpy_layer_idx():
    config = _heterogeneous_config()
    attention = LayerAttention(numpy.int64(1), num_kv_heads=2, head_dim=32)
    attention.kv_cache = MockKVCache(config)
    scheme = _attention_scheme(strategy=QuantizationStrategy.ATTN_HEAD)

    initialize_attn_qparams(attention, scheme, force_zero_point=False)

    assert attention.k_scale.shape == (2, 1, 1)


@pytest.mark.parametrize(
    "strategy,expected_shape,warns",
    [
        (QuantizationStrategy.TENSOR, (1,), False),
        (QuantizationStrategy.ATTN_HEAD, (8, 1, 1), True),
    ],
)
def test_initialize_attn_qparams_reports_kv_heads_mismatch(
    monkeypatch, strategy, expected_shape, warns
):
    # the config reports 8 key/value heads but the module projects to 2, as for
    # gemma-4 full-attention layers on transformers<5.15
    config = SimpleNamespace(
        num_attention_heads=16, num_key_value_heads=8, head_dim=256
    )
    attention = LayerAttention(0, num_kv_heads=2, head_dim=512)
    attention.kv_cache = MockKVCache(config)
    warning = Mock()
    monkeypatch.setattr(initialize_lifecycle.logger, "warning", warning)

    initialize_attn_qparams(
        attention, _attention_scheme(strategy=strategy), force_zero_point=False
    )

    # shapes keep following the config, so existing checkpoints still load
    assert attention.k_scale.shape == expected_shape
    if warns:
        warning.assert_called_once()
        assert "k_proj projects to 2 heads" in warning.call_args.args[0]
        assert warning.call_args.kwargs == {"log_once": True}
    else:
        warning.assert_not_called()


def test_initialize_attn_qparams_heterogeneous_kv_heads_mismatch():
    # a per-layer config which disagrees with its module is an error
    config = _heterogeneous_config()
    attention = LayerAttention(1, num_kv_heads=4, head_dim=32)
    attention.kv_cache = MockKVCache(config)
    scheme = _attention_scheme(strategy=QuantizationStrategy.ATTN_HEAD)

    with pytest.raises(ValueError, match="k_proj projects to 4 heads"):
        initialize_attn_qparams(attention, scheme, force_zero_point=False)


@pytest.mark.parametrize(
    "args",
    [
        dict(strategy=QuantizationStrategy.GROUP, group_size=8),
        dict(
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=16,
            num_bits=4,
            type="float",
        ),
    ],
)
def test_initialize_attn_qparams_rejects_static_group(args):
    config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, head_dim=16)
    attention = LayerAttention(0, num_kv_heads=2, head_dim=16)
    attention.kv_cache = MockKVCache(config)
    scheme = _attention_scheme(dynamic=False, **args)

    with pytest.raises(ValueError, match="not supported for attention"):
        initialize_attn_qparams(attention, scheme, force_zero_point=False)


@pytest.mark.parametrize(
    "args",
    [
        dict(strategy=QuantizationStrategy.GROUP, group_size=8, dynamic=True),
        dict(
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=16,
            num_bits=4,
            type="float",
            dynamic="local",
        ),
    ],
)
def test_initialize_attn_qparams_allows_dynamic_group(args):
    config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, head_dim=16)
    attention = LayerAttention(0, num_kv_heads=2, head_dim=16)
    attention.kv_cache = MockKVCache(config)

    initialize_attn_qparams(
        attention, _attention_scheme(**args), force_zero_point=False
    )

    assert not hasattr(attention, "k_scale")


def test_initialize_attn_qparams_heterogeneous_query_heads():
    # query scales follow the number of attention heads of the module's layer
    config = _heterogeneous_config()
    scheme = _attention_scheme(strategy=QuantizationStrategy.ATTN_HEAD)

    for layer_idx, expected_shape in enumerate([(8, 1, 1), (4, 1, 1)]):
        layer_config = config.per_layer_config[layer_idx]
        attention = LayerAttention(
            layer_idx, layer_config.num_key_value_heads, layer_config.head_dim
        )
        attention.kv_cache = MockKVCache(config)
        setattr(attention, IMPL_ATTR, torch.nn.Identity())

        initialize_attn_qparams(attention, scheme, force_zero_point=False)

        assert attention.q_scale.shape == expected_shape


def test_initialize_attn_qparams_homogeneous_transformers_config():
    # homogeneous transformers configs are read directly, without a layer_idx,
    # also on transformers>=5.14 where every config has a per_layer_config view
    config = PretrainedConfig(num_attention_heads=8, num_key_value_heads=2, head_dim=16)

    attention = _initialize_attn_qparams(config)

    assert attention.k_scale.shape == (2, 1, 1)


def test_initialize_attn_qparams_heterogeneous_partial_kv_heads():
    # a k_proj which is not a whole number of heads is not compared to the config
    config = _heterogeneous_config()
    attention = LayerAttention(0, num_kv_heads=4, head_dim=16)
    attention.k_proj = torch.nn.Linear(64, 40)
    attention.kv_cache = MockKVCache(config)
    scheme = _attention_scheme(strategy=QuantizationStrategy.ATTN_HEAD)

    initialize_attn_qparams(attention, scheme, force_zero_point=False)

    assert attention.k_scale.shape == (4, 1, 1)


@pytest.mark.parametrize(
    "weights,input_activations",
    [
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            None,
        ),
        (
            None,
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
    ],
)
def test_initialize_module_for_quantization(
    create_quantization_scheme, weights, input_activations, layer
):
    quantization_scheme = create_quantization_scheme(
        targets=["*"],
        weights=weights,
        input_activations=input_activations,
    )

    assert not hasattr(layer, "quantization_scheme")
    assert not hasattr(layer, "quantization_status")

    # add attributes, zero_points and scale
    initialize_module_for_quantization(layer, quantization_scheme)

    registered_params = {"weight", "bias"}
    if weights is not None:
        registered_params.add("weight_scale")
        registered_params.add("weight_zero_point")

    if input_activations is not None:
        registered_params.add("input_scale")
        registered_params.add("input_zero_point")

    for key in layer.state_dict().keys():
        assert key in registered_params
        registered_params.remove(key)

    assert len(registered_params) == 0

    assert hasattr(layer, "quantization_scheme")
    assert hasattr(layer, "quantization_status")

    assert layer.quantization_status == QuantizationStatus.INITIALIZED


@requires_gpu
@pytest.mark.parametrize(
    "weights,input_activations",
    [
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            None,
        ),
        (
            None,
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
        (
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
            QuantizationArgs(num_bits=NUM_BITS, symmetric=True),
        ),
    ],
)
def test_initialize_module_for_quantization_offloaded(
    create_quantization_scheme, weights, input_activations, layer
):
    set_onload_device(layer, "cuda:0")

    test_initialize_module_for_quantization(
        create_quantization_scheme,
        weights,
        input_activations,
        layer,
    )


@pytest.mark.parametrize(
    "weights,input_activations",
    [
        (
            QuantizationArgs(strategy="tensor"),
            QuantizationArgs(strategy="tensor"),
        ),
        (
            QuantizationArgs(strategy="channel"),
            None,
        ),
        (
            QuantizationArgs(strategy="group", group_size=2),
            None,
        ),
        (
            QuantizationArgs(strategy="group", group_size=2, actorder="weight"),
            None,
        ),
        (
            QuantizationArgs(
                strategy="tensor_group",
                group_size=16,
                type="float",
                num_bits=4,
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=FP8_E4M3_DATA.dtype,
            ),
            None,
        ),
        (
            QuantizationArgs(
                strategy="tensor_group",
                group_size=16,
                type="float",
                num_bits=4,
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=FP8_E4M3_DATA.dtype,
            ),
            QuantizationArgs(
                strategy="tensor_group",
                group_size=16,
                type="float",
                num_bits=4,
                dynamic="local",
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=FP8_E4M3_DATA.dtype,
            ),
        ),
        (
            QuantizationArgs(strategy="block", block_structure=[2, 4]),
            None,
        ),
    ],
)
def test_initialize_quantization_parameters(weights, input_activations):
    quantization_scheme = QuantizationScheme(
        targets=["*"],
        weights=weights,
        input_activations=input_activations,
    )
    layer = Linear(7, 8)
    initialize_module_for_quantization(layer, quantization_scheme)

    for q_type in ("input_activations", "weights"):
        args = getattr(quantization_scheme, q_type)
        if args is None:
            continue
        q_param_name = Q_PARAM_NAMES[q_type]

        if args.strategy == QuantizationStrategy.TENSOR_GROUP:
            if q_type == "weights":
                assert hasattr(layer, "weight_global_scale")
                assert layer.weight_global_scale.dtype == torch.float32
                assert layer.weight_global_scale.numel() == 1
                assert layer.weight_scale.dtype == layer.weight.dtype
            elif q_type == "input_activations":
                assert hasattr(layer, "input_global_scale")
                assert layer.input_global_scale.dtype == torch.float32
                assert layer.input_global_scale.numel() == 1
        else:
            assert not hasattr(layer, "weight_global_scale")
            assert not hasattr(layer, "input_global_scale")

        # scale and zero point
        if args.strategy == QuantizationStrategy.TENSOR:
            expected_shape = (1,)

        elif args.strategy == QuantizationStrategy.CHANNEL:  # only weight
            expected_shape = (layer.weight.shape[0], 1)

        elif args.strategy in (
            QuantizationStrategy.TENSOR_GROUP,
            QuantizationStrategy.GROUP,
        ):
            num_groups = math.ceil(layer.weight.shape[1] / args.group_size)
            expected_shape = (layer.weight.shape[0], max(num_groups, 1))

        elif args.strategy == QuantizationStrategy.BLOCK:
            # For block quantization, only weights get block-level scales
            # Activations fall back to tensor-level since shape is unknown at init
            if q_type == "weights" and args.block_structure is not None:
                block_height, block_width = args.block_structure
                rows, cols = layer.weight.shape[-2], layer.weight.shape[-1]
                num_rows_blocks = math.ceil(rows / block_height)
                num_cols_blocks = math.ceil(cols / block_width)
                expected_shape = (num_rows_blocks, num_cols_blocks)
            else:
                # For activations or when block_structure is None
                expected_shape = (1,)

        if not args.dynamic:
            assert getattr(layer, f"{q_param_name}_scale").shape == expected_shape
            assert getattr(layer, f"{q_param_name}_zero_point").shape == expected_shape
