# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from copy import deepcopy

import pytest
import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
)
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.transform.factory.base import TransformBase
from pydantic import ValidationError
from transformers import CompressedTensorsConfig, LlamaConfig, LlamaForCausalLM


def _config(order="offline-first", type="hadamard", precision=torch.float32):
    online = TransformArgs(targets=["Linear"], location="input")
    offline = TransformArgs(targets=["Linear"], location="weight_input", inverse=True)
    return TransformConfig(
        config_groups={
            "rotation": TransformScheme(
                type=type,
                precision=precision,
                randomize=True,
                apply=[offline, online]
                if order == "offline-first"
                else [online, offline],
            )
        }
    )


def _restore(model, metadata):
    if isinstance(metadata, TransformConfig):
        metadata = metadata.model_dump()
    apply_quantization_config(
        model, QuantizationConfig(config_groups={}, transform_config=metadata)
    )


def _linear_model():
    return torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16))


def _transforms(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TransformBase)
    }


def _tiny_llama():
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=32,
        )
    ).eval()
    model.set_attn_implementation("sdpa")
    return model


@pytest.mark.parametrize("order", ["offline-first", "online-first"])
@pytest.mark.parametrize("type", ["hadamard", "random-hadamard", "random-matrix"])
@pytest.mark.parametrize("precision", [torch.float32, torch.float64])
def test_restore_preserves_construction_order(order, type, precision):
    torch.manual_seed(123)
    original = _linear_model()
    restored = deepcopy(original)
    config = _config(order, type, precision)
    apply_transform_config(original, config)
    for source, target in zip(original, restored):
        target.weight.data.copy_(source.weight)
        target.bias.data.copy_(source.bias)
    fused = deepcopy(restored.state_dict())

    _restore(restored, config)

    # Restoration must not fuse the offline weight transform a second time.
    for name, tensor in fused.items():
        assert torch.equal(restored.state_dict()[name], tensor)
    expected_transforms = _transforms(original)
    actual_transforms = _transforms(restored)
    assert actual_transforms.keys() == expected_transforms.keys()
    for name, expected in expected_transforms.items():
        actual = actual_transforms[name]
        assert actual.weight.dtype == expected.weight.dtype
        assert torch.equal(actual.weight, expected.weight)
        if getattr(expected, "perm", None) is not None:
            assert torch.equal(actual.perm, expected.perm)
    matrices = list(actual_transforms.values())
    assert matrices[0].weight is matrices[1].weight

    # Saved tensors, including a changed matrix, are authoritative over RNG replay.
    with torch.no_grad():
        next(iter(expected_transforms.values())).weight[0, 0] += 0.25
    restored.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(3, 16)
    torch.testing.assert_close(restored(inputs), original(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("order", ["offline-first", "online-first"])
@pytest.mark.parametrize("type", ["random-hadamard", "random-matrix"])
def test_restore_preserves_multiple_size_rng_and_aliases(order, type):
    torch.manual_seed(123)
    original = torch.nn.Sequential(
        torch.nn.Linear(16, 8), torch.nn.Linear(8, 16), torch.nn.Linear(16, 16)
    )
    restored = deepcopy(original)
    config = _config(order, type)
    apply_transform_config(original, config)
    _restore(restored, config)
    for name, transform in _transforms(original).items():
        actual = restored.get_submodule(name)
        assert actual.weight.dtype == transform.weight.dtype
        assert torch.equal(actual.weight, transform.weight)
        if getattr(transform, "perm", None) is not None:
            assert torch.equal(actual.perm, transform.perm)
    assert restored[0].rotation_input.weight is restored[2].rotation_input.weight
    assert restored[0].rotation_input.weight is not restored[1].rotation_input.weight
    restored.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(3, 16)
    torch.testing.assert_close(restored(inputs), original(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("order", ["control", "online-first", "offline-first"])
def test_hf_reload_preserves_calibrated_transform_precision(tmp_path, order):
    torch.manual_seed(123)
    model = _tiny_llama()
    inputs = torch.tensor([[1, 2, 3, 4]])
    apply_quantization_config(
        model,
        QuantizationConfig(
            config_groups={
                "linear": QuantizationScheme(
                    targets=["Linear"],
                    input_activations=QuantizationArgs(
                        num_bits=8, type="int", strategy="token", dynamic=True
                    ),
                )
            },
            quantization_status=QuantizationStatus.FROZEN,
        ),
    )
    if order != "control":
        config = _config(order)
        for arg in config.config_groups["rotation"].apply:
            arg.targets = ["re:.*o_proj$"]
        apply_transform_config(model, config)
    with torch.no_grad():
        expected = model(inputs).logits
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
    ), info
    for name, transform in _transforms(model).items():
        assert restored.get_submodule(name).weight.dtype == transform.weight.dtype
    with torch.no_grad():
        actual = restored(inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_composed_configs_preserve_checkpoint_execution_order(
    tmp_path, sequential, overlap
):
    torch.manual_seed(123)
    model = _tiny_llama()
    apply_quantization_config(
        model,
        QuantizationConfig(
            config_groups={
                "linear": QuantizationScheme(
                    targets=["Linear"],
                    input_activations=QuantizationArgs(
                        num_bits=8, type="int", strategy="token", dynamic=True
                    ),
                )
            },
            quantization_status=QuantizationStatus.FROZEN,
        ),
    )
    groups = {}
    for index, (name, type) in enumerate(
        [("z_first", "hadamard"), ("a_second", "random-hadamard")]
    ):
        scheme = _config(type=type).config_groups["rotation"]
        for arg in scheme.apply:
            arg.targets = [
                "re:.*o_proj$" if overlap else f"re:.*layers\\.{index}\\..*o_proj$"
            ]
        groups[name] = scheme
        if sequential:
            apply_transform_config(model, TransformConfig(config_groups={name: scheme}))
    if not sequential:
        apply_transform_config(model, TransformConfig(config_groups=groups))
    assert tuple(model.transform_config.config_groups) == tuple(groups)
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(inputs).logits
    compressor = ModelCompressor.from_pretrained_model(
        model, quantization_format="dense"
    )
    model.save_pretrained(tmp_path)
    compressor.update_config(str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text())["quantization_config"][
        "transform_config"
    ]
    assert tuple(saved["config_groups"]) == tuple(groups)
    restored, info = LlamaForCausalLM.from_pretrained(
        tmp_path, attn_implementation="sdpa", output_loading_info=True
    )
    assert not any(
        info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), info
    with torch.no_grad():
        actual = restored(inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
    # The completion record must keep both sets of hooks after composition.
    model.get_submodule("model.layers.0.self_attn.o_proj")._forward_pre_hooks.clear()
    with pytest.raises(ValueError, match="modules or hooks are missing"):
        _restore(model, saved)
    del restored.get_submodule("model.layers.0.self_attn.o_proj").z_first_input
    with pytest.raises(ValueError, match="modules or hooks are missing"):
        _restore(restored, saved)


@pytest.mark.parametrize("same_config", [False, True])
def test_reusing_transform_group_name_fails_before_mutation(same_config):
    model = _linear_model()
    config = _config()
    apply_transform_config(model, config)
    before = deepcopy(model.state_dict())
    hooks = [tuple(layer._forward_pre_hooks) for layer in model]
    second = config if same_config else _config(type="random-hadamard")
    with pytest.raises(ValueError, match="already applied"):
        apply_transform_config(model, second)
    assert [tuple(layer._forward_pre_hooks) for layer in model] == hooks
    for name, tensor in before.items():
        assert torch.equal(model.state_dict()[name], tensor)


def _metadata(online=False):
    config = _config().model_dump()
    group = config["config_groups"]["rotation"]
    if not online:
        group["apply"] = [arg for arg in group["apply"] if arg["location"] != "input"]
    return config


def _wrapper(metadata):
    return CompressedTensorsConfig(
        config_groups={"W8A8": ["Linear"]}, transform_config=metadata
    )


@pytest.mark.parametrize("extra_location", ["scheme", "argument"])
def test_unknown_offline_annotations_survive_compressor_roundtrip(
    tmp_path, extra_location
):
    metadata = _metadata()
    group = metadata["config_groups"]["rotation"]
    target = group if extra_location == "scheme" else group["apply"][0]
    target["future_annotation"] = {"value": 1}
    original = deepcopy(metadata)

    compressor = ModelCompressor.from_compression_config(_wrapper(metadata))
    model = _linear_model()
    before = deepcopy(model.state_dict())
    _restore(model, compressor.quantization_config.transform_config)
    assert not _transforms(model)
    for name, tensor in before.items():
        assert torch.equal(model.state_dict()[name], tensor)
    compressor.update_config(str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["quantization_config"]["transform_config"] == original
    assert metadata == original

    rebuilt = ModelCompressor.from_pretrained_model(model)
    rebuilt.update_config(str(tmp_path))
    assert (
        json.loads((tmp_path / "config.json").read_text())["quantization_config"][
            "transform_config"
        ]
        == original
    )


@pytest.mark.parametrize("invalid", ["extra", "location", "precision"])
def test_invalid_online_metadata_is_rejected_before_installation(invalid):
    metadata = _metadata(online=True)
    group = metadata["config_groups"]["rotation"]
    if invalid == "extra":
        group["future_execution_mode"] = True
    elif invalid == "location":
        group["apply"][0]["location"] = "unknown"
    else:
        group["precision"] = "invalid_dtype"
    model = _linear_model()
    before = deepcopy(model.state_dict())
    with pytest.raises((ValidationError, ValueError)):
        _restore(model, metadata)
    assert not _transforms(model)
    assert not hasattr(model, "transform_config")
    assert all(not layer._forward_pre_hooks for layer in model)
    for name, tensor in before.items():
        assert torch.equal(model.state_dict()[name], tensor)


def test_offline_annotations_do_not_hide_executable_groups(tmp_path):
    metadata = _metadata()
    metadata["config_groups"]["rotation"]["future_annotation"] = True
    metadata["config_groups"]["runtime"] = _metadata(online=True)["config_groups"][
        "rotation"
    ]
    compressor = ModelCompressor.from_compression_config(_wrapper(metadata))
    model = _linear_model()
    _restore(model, compressor.quantization_config.transform_config)
    assert set(_transforms(model)) == {"0.runtime_input", "1.runtime_input"}
    assert model[0].runtime_input.weight.dtype == torch.float64
    compressor.update_config(str(tmp_path))
    assert (
        json.loads((tmp_path / "config.json").read_text())["quantization_config"][
            "transform_config"
        ]
        == metadata
    )


@pytest.mark.parametrize("source", ["nested", "outer", "both"])
def test_compressor_metadata_has_one_authoritative_source(source):
    metadata = _metadata(online=True)
    wrapper = _wrapper(metadata if source != "outer" else None)
    if source != "nested":
        wrapper.transform_config = deepcopy(metadata)
    compressor = ModelCompressor.from_compression_config(wrapper)
    assert compressor.transform_config.model_dump() == metadata
    assert compressor.quantization_config.transform_config == metadata


def test_conflicting_compressor_metadata_is_rejected():
    wrapper = _wrapper(_metadata(online=True))
    wrapper.transform_config = _metadata(online=False)
    with pytest.raises(ValueError, match="Conflicting transform configs"):
        ModelCompressor.from_compression_config(wrapper)


@pytest.mark.parametrize("consumer", ["restore", "compressor"])
def test_transform_group_order_is_part_of_config_identity(consumer):
    config = TransformConfig(
        config_groups={
            "first": _config().config_groups["rotation"],
            "second": _config(type="random-hadamard").config_groups["rotation"],
        }
    )
    reversed_config = config.model_dump()
    reversed_config["config_groups"] = dict(
        reversed(tuple(reversed_config["config_groups"].items()))
    )
    if consumer == "restore":
        model = _linear_model()
        _restore(model, config)
        with pytest.raises(ValueError, match="different transform config"):
            _restore(model, reversed_config)
    else:
        wrapper = _wrapper(config.model_dump())
        wrapper.transform_config = reversed_config
        with pytest.raises(ValueError, match="Conflicting transform configs"):
            ModelCompressor.from_compression_config(wrapper)


@pytest.mark.parametrize("initial_marker", [False, True])
def test_restore_is_idempotent_and_handles_empty_metadata_attribute(initial_marker):
    model = _linear_model()
    if initial_marker:
        model.transform_config = None
    config = _config()
    _restore(model, config)
    assert len(_transforms(model)) == 2
    first_hooks = [tuple(layer._forward_pre_hooks) for layer in model]
    inputs = torch.randn(2, 16)
    expected = model(inputs)
    _restore(model, config)
    assert [tuple(layer._forward_pre_hooks) for layer in model] == first_hooks
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("removed", ["module", "hook"])
def test_restore_detects_incomplete_existing_installation(removed):
    model = _linear_model()
    config = _config()
    _restore(model, config)
    if removed == "module":
        del model[0].rotation_input
    else:
        model[0]._forward_pre_hooks.clear()
    with pytest.raises(ValueError, match="modules or hooks are missing"):
        _restore(model, config)


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_failed_restore_rolls_back_and_can_retry(monkeypatch, attention, exception):
    model = _tiny_llama() if attention else _linear_model()
    config = _config()
    if attention:
        config.config_groups["rotation"] = TransformScheme(
            type="hadamard",
            head_dim=8,
            apply=[
                TransformArgs(targets=["LlamaAttention"], location=location)
                for location in ("q_attn", "k_cache")
            ],
        )
    before = deepcopy(model.state_dict())
    modules_before = tuple(model.named_modules())
    original_apply = TransformFactory._apply_to_module
    count = 0

    def failing_apply(self, model, module, args):
        nonlocal count
        count += 1
        if count == 2:
            raise exception("second installation failed")
        original_apply(self, model, module, args)

    with monkeypatch.context() as patch:
        patch.setattr(TransformFactory, "_apply_to_module", failing_apply)
        with pytest.raises(exception, match="second installation failed"):
            _restore(model, config)
    assert tuple(model.named_modules()) == modules_before
    assert not hasattr(model, "transform_config")
    assert all(not module._forward_pre_hooks for _, module in modules_before)
    if attention:
        assert model.config._attn_implementation == "sdpa"
    for name, tensor in before.items():
        assert torch.equal(model.state_dict()[name], tensor)

    _restore(model, config)
    assert len(_transforms(model)) == (4 if attention else 2)
