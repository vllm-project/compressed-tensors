# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest
import torch
from compressed_tensors.offload import offload_module
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationMetadata,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
)
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture(autouse=True)
def deterministic_empty_parameters():
    enabled = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    yield
    torch.use_deterministic_algorithms(enabled)
    torch.utils.deterministic.fill_uninitialized_memory = fill


def _model():
    torch.manual_seed(1234)
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128,
        )
    ).eval()
    model.set_attn_implementation("sdpa")
    return model


def _args(local=False, static=False):
    return QuantizationArgs(
        dynamic="local" if local else not static,
        strategy="tensor_group" if local else "tensor" if static else "group",
        group_size=16 if local else None if static else 8,
        num_bits=4 if local else 8,
        type="float" if local else "int",
    )


def _config(args, *, query=True, global_args=None):
    return QuantizationConfig(
        config_groups=(
            {
                "attention": QuantizationScheme(
                    targets=["LlamaAttention"], input_activations=args
                )
            }
            if query
            else {}
        ),
        kv_cache_scheme=global_args if query else args,
        quantization_status=QuantizationStatus.FROZEN,
    )


def _offload(module, device, tmp_path):
    if device is not None:
        offload_module(
            module,
            "cpu",
            device,
            **({"offload_dir": str(tmp_path)} if device == "disk" else {}),
        )


@pytest.mark.parametrize("offload", [None, "cpu", "disk"])
@pytest.mark.parametrize(
    "transition", ["global_override", "local_to_dynamic", "query_off"]
)
def test_attention_reinitialization_matches_fresh_scheme(transition, offload, tmp_path):
    model = _model()
    reference = deepcopy(model)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    for layer in model.model.layers:
        attention = layer.self_attn
        for base in ("q", "k", "v"):
            attention.register_buffer(f"{base}_shape", torch.tensor([2, 4]))
        _offload(attention, offload, tmp_path)
    query = transition != "query_off"
    final = _config(_args(), query=query)
    if transition == "global_override":
        apply_quantization_config(
            model, _config(_args(), global_args=_args(local=True))
        )
    else:
        apply_quantization_config(model, _config(_args(local=query, static=not query)))
        apply_quantization_config(model, final)
    apply_quantization_config(reference, final)
    inputs = torch.tensor([[1, 3, 7, 2, 8, 5]])
    with torch.no_grad():
        actual = model(inputs, use_cache=False).logits
        expected = reference(inputs, use_cache=False).logits
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for layer in model.model.layers:
        attention = layer.self_attn
        for base in ("q", "k", "v"):
            for suffix in ("global_scale", "scale", "zero_point"):
                assert not hasattr(attention, f"{base}_{suffix}")
            assert torch.equal(
                getattr(attention, f"{base}_shape"), torch.tensor([2, 4])
            )
    for name, value in weights.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("offload", [None, "cpu", "disk"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("clear", ["clear_all_qparams", "clear_quantization"])
def test_clear_attention_metadata_preserves_native_layout(
    clear, partial, offload, tmp_path
):
    model = _model()
    attention = model.model.layers[0].self_attn
    weight = attention.q_proj.weight.detach().clone()
    names = [
        f"{base}_{suffix}"
        for base in ("q", "k", "v")
        for suffix in ("global_scale", "scale", "zero_point")
    ]
    if partial:
        names = ["q_global_scale", "k_zero_point", "v_scale"]
    for name in names:
        dtype = torch.int8 if name.endswith("zero_point") else torch.float32
        attention.register_parameter(
            name, torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        )
    for base in ("q", "k", "v"):
        attention.register_buffer(f"{base}_shape", torch.tensor([2, 4]))
    attention.quantization_scheme = _config(_args()).config_groups["attention"]
    _offload(attention, offload, tmp_path)
    getattr(QuantizationMetadata, clear)(attention)
    for name in names:
        assert not hasattr(attention, name)
    for base in ("q", "k", "v"):
        assert torch.equal(getattr(attention, f"{base}_shape"), torch.tensor([2, 4]))
    assert torch.equal(attention.q_proj.weight, weight)
    assert hasattr(attention, "quantization_scheme") == (clear == "clear_all_qparams")
