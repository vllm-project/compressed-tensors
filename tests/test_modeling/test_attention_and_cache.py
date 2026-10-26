# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.modeling import (
    IMPL_ATTR,
    KV_CACHE_ATTR,
    QuantizedAttentionImpl,
    QuantizedKVCache,
    initialize_hooked_attention,
    initialize_hooked_kv_cache,
    register_key_hook,
    register_query_hook,
    register_value_hook,
)
from compressed_tensors.modeling.attention import _hooked_attention
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    apply_quantization_config,
    disable_quantization,
)
from tests.testing_utils import requires_gpu
from transformers import (
    AutoModelForCausalLM,
    CLIPVisionConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlavaConfig,
    LlavaForConditionalGeneration,
)
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@requires_gpu
def test_attention_cache():
    model = AutoModelForCausalLM.from_pretrained(
        "nm-testing/llama2.c-stories15M", device_map="cuda"
    )
    inputs = {key: value.to("cuda") for key, value in model.dummy_inputs.items()}
    true_outputs = model(**inputs)
    layers = model.model.layers

    # check if hooks work
    k_called = [False for _ in range(len(layers))]
    v_called = [False for _ in range(len(layers))]

    # apply kv cache quantization
    _apply_kv_cache(model, layers, k_called, v_called)

    # check kv cache quantization
    outputs = model(**inputs)
    assert torch.equal(outputs.logits, true_outputs.logits)
    assert all(k_called) and all(v_called)

    """ apply attention quantization after kv cache quantization """

    # check if hooks work
    q_called = [False for _ in range(len(layers))]
    k_called = [False for _ in range(len(layers))]
    v_called = [False for _ in range(len(layers))]

    # apply attention quantization
    _apply_attention(model, layers, q_called, k_called, v_called)

    # check attention quantization
    outputs = model(**inputs)
    assert torch.equal(outputs.logits, true_outputs.logits)
    assert all(q_called) and all(k_called) and all(v_called)


def _apply_kv_cache(model, layers, k_called, v_called):
    for layer_index, layer in enumerate(layers):
        module = layer.self_attn
        initialize_hooked_kv_cache(model, module)
        assert isinstance(getattr(module, KV_CACHE_ATTR), QuantizedKVCache)

        # reapply is no-op
        initialize_hooked_kv_cache(model, module)

        def k_hook(_module, _states, layer_index=layer_index):  # NOTE: capture by value
            k_called[layer_index] = True

        def v_hook(_module, _states, layer_index=layer_index):
            my_index = layer_index
            v_called[my_index] = True

        register_key_hook(module, k_hook)
        register_value_hook(module, v_hook)


def _apply_attention(model, layers, q_called, k_called, v_called):
    for layer_index, layer in enumerate(layers):
        module = layer.self_attn
        initialize_hooked_attention(model, module)
        assert isinstance(getattr(module, IMPL_ATTR), QuantizedAttentionImpl)

        # reapply is no-op
        initialize_hooked_attention(model, module)

        def q_hook(_module, _states, layer_index=layer_index):
            q_called[layer_index] = True

        def k_hook(_module, _states, layer_index=layer_index):
            k_called[layer_index] = True

        def v_hook(_module, _states, layer_index=layer_index):
            v_called[layer_index] = True

        register_query_hook(module, q_hook)
        register_key_hook(module, k_hook)
        register_value_hook(module, v_hook)


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_attention_quantization_skips_attention_that_cannot_be_quantized(
    attn_implementation,
):
    # attention modules without a cache argument, like the vision tower of llava,
    # can't be quantized, so they keep the original attention implementation
    model, inputs = _tiny_llava()
    model.set_attn_implementation(attn_implementation)
    with torch.no_grad():
        true_logits = model(**inputs).logits

    apply_quantization_config(
        model, _attention_config(["re:.*language_model.*self_attn$"])
    )
    attention = {
        name: module
        for name, module in model.named_modules()
        if name.endswith("self_attn")
    }
    text = [m for name, m in attention.items() if "language_model" in name]
    vision = [m for name, m in attention.items() if "vision_tower" in name]
    assert len(text) == len(vision) == 2
    assert all(hasattr(module, IMPL_ATTR) for module in text)
    assert not any(hasattr(module, IMPL_ATTR) for module in vision)
    for module in text:
        for name in ("q", "k", "v"):
            getattr(module, f"{name}_scale").data.fill_(1.0)

    # the vision tower uses the hooked attention too, without an impl
    vision_backend = model.config.vision_config._attn_implementation
    assert ALL_ATTENTION_FUNCTIONS[vision_backend] is _hooked_attention
    assert (
        ALL_MASK_ATTENTION_FUNCTIONS[vision_backend]
        is ALL_MASK_ATTENTION_FUNCTIONS[attn_implementation]
    )

    query_modules = []

    def query_hook(module, _query):
        query_modules.append(module)

    for module in text:
        register_query_hook(module, query_hook)

    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.isfinite(logits).all()
    assert query_modules == text

    # without quantization, the outputs match the original model
    model.apply(disable_quantization)
    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.equal(logits, true_logits)


def test_attention_quantization_of_some_layers():
    # attention modules with key/value projections and a cache argument can be
    # quantized, so one which the scheme does not target raises instead of silently
    # keeping the original implementation
    model, input_ids = _tiny_llama()
    apply_quantization_config(
        model, _attention_config(["re:.*layers\\.0\\.self_attn$"])
    )

    with torch.no_grad():
        with pytest.raises(ValueError, match="has no impl submodule"):
            model(input_ids=input_ids)


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_attention_quantization_without_impl(dynamic, attn_implementation):
    # an attention module which can be quantized and lost its impl raises instead of
    # running without quantizing its query
    model, input_ids = _tiny_llama()
    model.set_attn_implementation(attn_implementation)
    apply_quantization_config(model, _attention_config(["LlamaAttention"], dynamic))
    delattr(model.model.layers[0].self_attn, IMPL_ATTR)

    with torch.no_grad():
        with pytest.raises(ValueError, match="has no impl submodule"):
            model(input_ids=input_ids)


def test_attention_quantization_replaced_module_without_impl():
    model, input_ids = _tiny_llama()
    attention = model.model.layers[0].self_attn
    replacement = type(attention)(model.config, layer_idx=0)
    replacement.load_state_dict(attention.state_dict())
    apply_quantization_config(model, _attention_config(["LlamaAttention"], True))
    model.model.layers[0].self_attn = replacement

    with torch.no_grad(), pytest.raises(ValueError, match="has no impl submodule"):
        model(input_ids=input_ids)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_value_cache_transform_preserves_attention(attn_implementation):
    from compressed_tensors.transform import (
        TransformArgs,
        TransformConfig,
        TransformScheme,
        apply_transform_config,
    )

    model, input_ids = _tiny_llama()
    model.set_attn_implementation(attn_implementation)
    with torch.no_grad():
        reference = model(input_ids=input_ids).logits
    scheme = TransformScheme(
        type="hadamard",
        head_dim=16,
        apply=[
            TransformArgs(targets=["LlamaAttention"], location="v_cache"),
            TransformArgs(
                targets=["re:.*o_proj$"], location="weight_input", inverse=True
            ),
        ],
    )
    config = TransformConfig(config_groups={"R2": scheme})
    config = TransformConfig.model_validate_json(config.model_dump_json())
    apply_transform_config(model, config)
    with torch.no_grad():
        actual = model(input_ids=input_ids).logits
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_query_transform_does_not_quantize_query_with_kv_only_scheme():
    from compressed_tensors.transform import (
        TransformArgs,
        TransformConfig,
        TransformScheme,
        apply_transform_config,
    )

    model, input_ids = _tiny_llama()
    args = QuantizationArgs(num_bits=8, type="float", strategy="tensor")
    apply_quantization_config(
        model, QuantizationConfig(config_groups={}, kv_cache_scheme=args)
    )
    for layer in model.model.layers:
        layer.self_attn.k_scale.data.fill_(1.0)
        layer.self_attn.v_scale.data.fill_(1.0)
    scheme = TransformScheme(
        type="hadamard",
        head_dim=16,
        apply=[
            TransformArgs(targets=["LlamaAttention"], location="q_attn"),
            TransformArgs(targets=["LlamaAttention"], location="k_cache"),
        ],
    )
    apply_transform_config(model, TransformConfig(config_groups={"R3": scheme}))
    seen = []
    for layer in model.model.layers:
        register_query_hook(layer.self_attn, lambda module, query: seen.append(module))
        assert not hasattr(layer.self_attn, "q_scale")
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    assert torch.isfinite(logits).all()
    assert seen == [layer.self_attn for layer in model.model.layers]


def test_kv_query_transform_quantizer_config_roundtrip():
    from compressed_tensors.transform import (
        TransformArgs,
        TransformConfig,
        TransformScheme,
        apply_transform_config,
    )
    from transformers.utils.quantization_config import CompressedTensorsConfig

    model, input_ids = _tiny_llama()
    args = QuantizationArgs(num_bits=8, type="float", strategy="tensor")
    apply_quantization_config(
        model, QuantizationConfig(config_groups={}, kv_cache_scheme=args)
    )
    for layer in model.model.layers:
        layer.self_attn.k_scale.data.fill_(1.0)
        layer.self_attn.v_scale.data.fill_(1.0)
    transforms = TransformConfig(
        config_groups={
            "R3": TransformScheme(
                type="hadamard",
                head_dim=16,
                apply=[
                    TransformArgs(targets=["LlamaAttention"], location="q_attn"),
                    TransformArgs(targets=["LlamaAttention"], location="k_cache"),
                ],
            ),
        }
    )
    apply_transform_config(model, transforms)
    quantization = QuantizationConfig.from_pretrained(model)
    assert not quantization.config_groups
    saved = quantization.model_dump()
    saved["transform_config"] = transforms.model_dump()
    wrapper = CompressedTensorsConfig(**saved)
    restored, _ = _tiny_llama()
    apply_quantization_config(restored, wrapper.quantization_config)
    restored.load_state_dict(model.state_dict(), strict=True)
    for layer in restored.model.layers:
        assert not hasattr(layer.self_attn, "q_scale")
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits
        actual = restored(input_ids=input_ids).logits
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("location", ["q_attn", "v_cache"])
def test_online_attention_transform_checkpoint_reload(tmp_path, location):
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.transform import (
        TransformArgs,
        TransformConfig,
        TransformScheme,
        apply_transform_config,
    )

    model, input_ids = _tiny_llama()
    if not hasattr(model, "all_tied_weights_keys"):
        pytest.skip("requires transformers tied-weight mapping support")
    args = QuantizationArgs(num_bits=8, type="float", strategy="tensor")
    apply_quantization_config(
        model, QuantizationConfig(config_groups={}, kv_cache_scheme=args)
    )
    for layer in model.model.layers:
        layer.self_attn.k_scale.data.fill_(1.0)
        layer.self_attn.v_scale.data.fill_(1.0)
    apply_transform_config(
        model,
        TransformConfig(
            config_groups={
                "online": TransformScheme(
                    type="hadamard",
                    head_dim=16,
                    apply=[
                        TransformArgs(targets=["LlamaAttention"], location=location),
                    ],
                ),
            }
        ),
    )
    with torch.no_grad():
        reference = model(input_ids=input_ids).logits
    compressor = ModelCompressor.from_pretrained_model(
        model, quantization_format="dense"
    )
    model.save_pretrained(tmp_path)
    compressor.update_config(str(tmp_path))
    restored, info = LlamaForCausalLM.from_pretrained(
        tmp_path, attn_implementation="sdpa", output_loading_info=True
    )
    assert not any(
        info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    with torch.no_grad():
        actual = restored(input_ids=input_ids).logits
    assert torch.equal(actual, reference)


def _attention_config(targets, dynamic=False):
    args = QuantizationArgs(
        num_bits=8, type="float", strategy="tensor", symmetric=True, dynamic=dynamic
    )
    scheme = QuantizationScheme(targets=targets, input_activations=args)
    return QuantizationConfig(config_groups={"attention": scheme})


def _tiny_llama():
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    model = LlamaForCausalLM(config).eval()
    model.set_attn_implementation("sdpa")
    return model, torch.randint(0, config.vocab_size, (1, 8))


def _tiny_llava():
    vision_config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
    )
    text_config = LlamaConfig(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=160,
    )
    config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=150,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
    )
    model = LlavaForConditionalGeneration(config).eval()
    model.set_attn_implementation("sdpa")

    # one image token per patch, the "default" strategy drops the CLS feature
    num_image_tokens = (32 // 8) ** 2
    input_ids = torch.cat(
        [torch.full((1, num_image_tokens), 150), torch.randint(0, 128, (1, 6))], dim=1
    )
    return model, {"input_ids": input_ids, "pixel_values": torch.randn(1, 3, 32, 32)}
