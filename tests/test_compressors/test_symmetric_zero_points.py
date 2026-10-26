# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest
import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.compressors.base import BaseCompressor, compress_module
from compressed_tensors.compressors.dense.base import DenseCompressor
from compressed_tensors.offload import offload_module
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
    disable_quantization,
    enable_quantization,
    initialize_module_for_quantization,
)
from transformers import LlamaConfig, LlamaForCausalLM


def _model():
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config).eval()


@pytest.mark.parametrize(
    "target,kind,symmetric",
    [
        ("attention", "float", True),
        ("attention", "int", True),
        ("attention", "int", False),
        ("kv", "float", True),
        ("kv", "int", False),
        ("linear", "int", True),
        ("linear", "int", False),
    ],
)
@pytest.mark.parametrize("offload_device", [None, "cpu", "disk"])
def test_symmetric_zero_point_checkpoint(
    target, kind, symmetric, offload_device, tmp_path
):
    model = _model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        original = model(input_ids).logits
    args = QuantizationArgs(
        num_bits=8, type=kind, strategy="tensor", symmetric=symmetric
    )
    if target == "kv":
        config = QuantizationConfig(config_groups={}, kv_cache_scheme=args)
    else:
        scheme = QuantizationScheme(
            targets=["LlamaAttention" if target == "attention" else "Linear"],
            input_activations=args,
        )
        config = QuantizationConfig(config_groups={"test": scheme})
    apply_quantization_config(model, config, show_progress=False)
    quantized = [m for m in model.modules() if hasattr(m, "quantization_scheme")]
    for index, module in enumerate(quantized):
        module.quantization_status = QuantizationStatus.FROZEN
        for name, param in module.named_parameters(recurse=False):
            if name.endswith("_scale"):
                param.data.fill_(0.1)
            elif name.endswith("_zero_point"):
                param.data.fill_(0 if symmetric else 3)
        if offload_device is not None:
            offload_dir = tmp_path / f"offload-{index}"
            offload_dir.mkdir()
            offload_module(
                module,
                onload_device="cpu",
                offload_device=offload_device,
                **(
                    {"offload_dir": str(offload_dir)}
                    if offload_device == "disk"
                    else {}
                ),
            )
    model.apply(disable_quantization)
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids).logits, original, rtol=0, atol=0)
    model.apply(enable_quantization)
    with torch.no_grad():
        expected = model(input_ids).logits
    assert torch.isfinite(expected).all()
    assert not torch.equal(expected, original)

    compressor = ModelCompressor.from_pretrained_model(model)
    compressor.compress_model(model)
    zero_points = {
        n: p.clone() for n, p in model.state_dict().items() if n.endswith("_zero_point")
    }
    if symmetric:
        assert not zero_points
    else:
        assert zero_points
        assert all(torch.equal(p, torch.full_like(p, 3)) for p in zero_points.values())
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    compressor.update_config(checkpoint)
    loaded, info = LlamaForCausalLM.from_pretrained(
        checkpoint, output_loading_info=True, attn_implementation="sdpa"
    )
    assert not any(
        info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids).logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(model(input_ids).logits, expected, rtol=0, atol=0)
    loaded.apply(disable_quantization)
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids).logits, original, rtol=0, atol=0)
    compressor.compress_model(model)
    assert set(n for n in model.state_dict() if n.endswith("_zero_point")) == set(
        zero_points
    )


@pytest.mark.parametrize("symmetric", [False, True])
def test_dense_compressor_standalone_identity(symmetric):
    scheme = QuantizationScheme(
        targets=["Linear"],
        input_activations=QuantizationArgs(symmetric=symmetric),
    )
    state = {"weight": torch.ones(2, 2), "input_zero_point": torch.zeros(1)}
    before = dict(state)
    assert DenseCompressor.compress(state, scheme) is state
    assert DenseCompressor.decompress(state, scheme) is state
    assert state.keys() == before.keys()
    assert all(state[key] is value for key, value in before.items())


@pytest.mark.parametrize("symmetric", [False, True])
def test_module_compression_preserves_custom_mapping_and_native_attention_names(
    symmetric,
):
    class RetainingCompressor(BaseCompressor):
        @classmethod
        def compress(cls, state_dict, scheme):
            cls.retained = state_dict
            cls.before = dict(state_dict)
            return state_dict

    module = torch.nn.Linear(4, 2)
    scheme = QuantizationScheme(
        targets=["Linear"],
        input_activations=QuantizationArgs(symmetric=symmetric),
    )
    initialize_module_for_quantization(module, scheme)
    module.input_scale.data.fill_(0.1)
    module.input_zero_point.data.fill_(0 if symmetric else 3)
    for name in ("q_zero_point", "k_zero_point", "v_zero_point"):
        module.register_buffer(name, torch.tensor([7]))
    weight = module.weight.detach().clone()
    RetainingCompressor.compress_module(module)
    assert RetainingCompressor.retained.keys() == RetainingCompressor.before.keys()
    assert all(
        RetainingCompressor.retained[key] is value
        for key, value in RetainingCompressor.before.items()
    )
    assert hasattr(module, "input_zero_point") is not symmetric
    for name in ("q_zero_point", "k_zero_point", "v_zero_point"):
        torch.testing.assert_close(getattr(module, name), torch.tensor([7]))
    torch.testing.assert_close(module.weight, weight, rtol=0, atol=0)


def test_no_activation_scheme_preserves_native_zero_points():
    module = torch.nn.Linear(4, 2)
    module.quantization_scheme = QuantizationScheme(targets=["Linear"])
    module.register_buffer("input_zero_point", torch.tensor([9]))
    before = deepcopy(module.state_dict())
    compress_module(module)
    assert module.state_dict().keys() == before.keys()
    for key, value in before.items():
        torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)
