# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest
import torch
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_FORMAT,
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.quantization.lifecycle.apply import apply_quantization_config
from compressed_tensors.quantization.quant_config import (
    _map_to_checkpoint_names,
    get_vllm_module_type,
)
from compressed_tensors.utils.match import match_targets
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
    assert config.moe_sparsity_config is None


def test_config_with_moe_sparsity():
    from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod

    config = QuantizationConfig(
        config_groups={"group_1": QuantizationScheme(targets=[])},
        moe_sparsity_config=ExpertSparsityConfig(
            method="prune", num_experts=[None, 129, 253]
        ),
    )
    assert isinstance(config.moe_sparsity_config, ExpertSparsityConfig)
    assert config.moe_sparsity_config.method == MoESparsityMethod.PRUNE

    # round-trips through dict (de)serialization
    restored = QuantizationConfig.model_validate(config.model_dump())
    assert isinstance(restored.moe_sparsity_config, ExpertSparsityConfig)
    assert restored.moe_sparsity_config.num_experts == [None, 129, 253]


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


@pytest.mark.parametrize("conversion", ["none", "prefix", "weight"])
@pytest.mark.parametrize(
    "targets",
    [["runtime.0"], [r"re:^runtime\."], ["Linear"], ["Linear", r"re:^runtime\."]],
)
def test_export_preserves_targets_and_expands_checkpoint_ignores(conversion, targets):
    WeightRenaming = pytest.importorskip(
        "transformers.core_model_loading"
    ).WeightRenaming
    model = torch.nn.Module()
    model.runtime = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(33)])
    if conversion == "prefix":
        model._weight_conversions = [WeightRenaming("checkpoint.", "runtime.")]
    elif conversion == "weight":
        model._weight_conversions = [
            WeightRenaming(f"checkpoint.{i}.weight", f"runtime.{i}.weight")
            for i in range(33)
        ]
    scheme = QuantizationScheme(targets=targets, weights=QuantizationArgs(num_bits=8))
    model.runtime[0].quantization_scheme = scheme
    model.runtime[0].quantization_status = QuantizationStatus.FROZEN
    original_scheme = scheme.model_dump()

    exported = QuantizationConfig.from_pretrained(model)
    prefix = "runtime" if conversion == "none" else "checkpoint"
    expected_ignores = [f"{prefix}.{i}" for i in range(1, 33)]
    assert exported.ignore == expected_ignores
    assert exported.model_dump()["ignore"] == expected_ignores
    assert exported.config_groups["group_0"].model_dump() == original_scheme
    assert scheme.model_dump() == original_scheme
    restored = QuantizationConfig.model_validate_json(exported.model_dump_json())
    assert restored.ignore == expected_ignores
    assert restored.config_groups["group_0"].targets == targets
    assert QuantizationConfig.from_pretrained(model) == exported


@pytest.mark.parametrize("layout", ["trailing", "nested", "flat"])
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_mtp_export_preserves_scheme_selection(layout, selective, shared):
    WeightRenaming = pytest.importorskip(
        "transformers.core_model_loading"
    ).WeightRenaming
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(12)])
    model.mtp = torch.nn.Module()
    model.mtp.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.mtp.layers[0].mtp_block = torch.nn.Module()
    for block in [*model.model.layers, model.mtp.layers[0].mtp_block]:
        block.mlp = torch.nn.Module()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            setattr(block.mlp, projection, torch.nn.Linear(4, 4))
        block.mlp.embedding = torch.nn.Embedding(4, 4)
    model.mtp.layers[0].eh_proj = torch.nn.Linear(8, 4)
    model.lm_head = torch.nn.Linear(4, 4)
    checkpoint_prefix = {
        "trailing": "model.layers.12.",
        "nested": "model.mtp.layers.0.transformer_block.",
        "flat": "model.mtp.layers.0.",
    }[layout]
    model._weight_conversions = [
        WeightRenaming(checkpoint_prefix, "mtp.layers.0.mtp_block."),
        WeightRenaming(
            (
                "model.mtp.layers.0.input_proj.weight"
                if layout != "trailing"
                else "model.layers.12.eh_proj.weight"
            ),
            "mtp.layers.0.eh_proj.weight",
        ),
    ]
    schemes = [
        QuantizationScheme(
            targets=(
                [r"re:^model\.layers\.(?![37]\.)\d+\.mlp\.(?:gate|up)_proj$"]
                if selective
                else [r"re:^model\.layers\."]
            ),
            weights=QuantizationArgs(num_bits=8),
        ),
        QuantizationScheme(
            targets=(
                [r"re:^mtp\..*(?<!down)_proj$"]
                if selective
                else [r"re:^mtp\..*mtp_block\."]
            ),
            weights=QuantizationArgs(num_bits=4),
        ),
    ]
    if shared:
        schemes = [
            schemes[0].model_copy(
                update={"targets": schemes[0].targets + schemes[1].targets}
            )
        ]
    original = [scheme.model_dump() for scheme in schemes]
    expected = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name.endswith("eh_proj"):
            continue
        for scheme in schemes:
            if match_targets(name, module, scheme.targets):
                module.quantization_scheme = scheme
                module.quantization_status = QuantizationStatus.FROZEN
                expected[name] = scheme.weights.num_bits
    exported = QuantizationConfig.from_pretrained(model)
    assert [scheme.model_dump() for scheme in schemes] == original
    assert all(not name.startswith("re:") for name in exported.ignore)
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            continue
        checkpoint_name = _map_to_checkpoint_names(model, [name])[0]
        serving_name = (
            checkpoint_name.replace("model.mtp.", "model.")
            if layout == "nested"
            else checkpoint_name
        )
        for alias in (name, checkpoint_name, serving_name):
            selected = [
                scheme.weights.num_bits
                for scheme in exported.config_groups.values()
                if match_targets(alias, module, scheme.targets)
            ]
            assert selected == ([expected[name]] if name in expected else [])
    assert QuantizationConfig.from_pretrained(model) == exported
    assert max(len(s.targets) for s in exported.config_groups.values()) == 1


@pytest.mark.parametrize("targets", [["Linear"], [r"re:.*\.proj$"]])
@pytest.mark.parametrize("prefix_collision", [False, True])
def test_mtp_export_preserves_excluded_types(targets, prefix_collision):
    WeightRenaming = pytest.importorskip(
        "transformers.core_model_loading"
    ).WeightRenaming

    class CustomLinear(torch.nn.Linear):
        pass

    model = torch.nn.Module()
    for prefix in ("model", "mtp", *(["shadow"] if prefix_collision else [])):
        branch = torch.nn.Module()
        branch.layers = torch.nn.ModuleList([torch.nn.Module()])
        setattr(model, prefix, branch)
        mlp = branch.layers[0].mlp = torch.nn.Module()
        if prefix == "shadow":
            mlp.proj = torch.nn.Embedding(8, 8)
        else:
            mlp.proj = torch.nn.Linear(8, 8)
            mlp.custom = CustomLinear(8, 8)
            mlp.embedding = torch.nn.Embedding(8, 8)
    model._weight_conversions = [WeightRenaming("model.layers.1.", "mtp.layers.0.")]
    restored = deepcopy(model)
    config = QuantizationConfig(
        config_groups={
            "shared": QuantizationScheme(
                targets=targets, weights=QuantizationArgs(num_bits=8)
            )
        },
        ignore=[r"re:^shadow\."],
    )
    apply_quantization_config(model, config, show_progress=False)
    exported = QuantizationConfig.from_pretrained(model)
    apply_quantization_config(restored, exported, show_progress=False)
    for name, module in model.named_modules():
        before = getattr(module, "quantization_scheme", None)
        after = getattr(restored.get_submodule(name), "quantization_scheme", None)
        assert (before is None) == (after is None), name
        if before is not None:
            assert before.weights == after.weights, name
    assert not exported.ignore  # Guards do not expand the serialized ignore list.
    assert len(exported.config_groups["group_0"].targets) == 1
    assert config.config_groups["shared"].targets == targets


def test_get_vllm_module_type():
    assert get_vllm_module_type("ExpertMLP") == "ExpertMLP"
    assert get_vllm_module_type("ExpertMLPWithGate") == "ExpertMLPWithGate"
    assert get_vllm_module_type("ExpertMLPWithoutGate") == "ExpertMLPWithoutGate"
    assert get_vllm_module_type("Linear") == "Linear"
    assert get_vllm_module_type("DeepseekV4TopKRouter") == "Linear"
    assert get_vllm_module_type("DeepseekV4HashRouter") == "Linear"
    assert get_vllm_module_type("JetMoeTopKGating") == "Linear"
    assert get_vllm_module_type("DeepseekV4TopKGate") == "Linear"
    assert get_vllm_module_type("Qwen3NextGatedDeltaNet") == "Qwen3NextGatedDeltaNet"
    assert get_vllm_module_type("Qwen3NextRMSNormGated") == "Qwen3NextRMSNormGated"


def test_quantization_config_merge():
    config = QuantizationConfig(
        config_groups={
            "config_group_0": QuantizationScheme(
                targets=["re:.*self_attn.*"],
                weights=QuantizationArgs(num_bits=4, symmetric=True, group_size=128),
            )
        },
        ignore=["lm_head", "model.layers.0.mlp.gate_proj", "re:.*mtp.*"],
        quantization_status=QuantizationStatus.INITIALIZED,
    )

    new_config = QuantizationConfig(
        config_groups={
            "config_group_0": QuantizationScheme(
                targets=["re:.*mlp.*"],
                weights=QuantizationArgs(num_bits=8, symmetric=False, group_size=128),
            )
        },
        ignore=["lm_head"],
        quantization_status=QuantizationStatus.COMPRESSED,
    )

    config.merge(new_config)

    ordered_schemes = list(config.config_groups.values())
    assert len(ordered_schemes) == 2
    assert ordered_schemes[0].targets[0] == "re:.*self_attn.*"
    assert ordered_schemes[1].targets[0] == "re:.*mlp.*"

    # should strip out "model.layers.0.mlp.gate_proj" from ignore
    assert set(config.ignore) == set(["lm_head", "re:.*mtp.*"])

    assert config.quantization_status == QuantizationStatus.COMPRESSED


def test_imatrix_mse_weight_observer_requires_calibration_data():
    from compressed_tensors.quantization import QuantizationArgs

    config = QuantizationConfig(
        config_groups={
            "group_1": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(observer="imatrix_mse"),
            )
        }
    )

    assert config.requires_calibration_data()


def test_default_weight_observer_does_not_require_calibration_data():
    from compressed_tensors.quantization import QuantizationArgs

    config = QuantizationConfig(
        config_groups={
            "group_1": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(),
            )
        }
    )

    assert not config.requires_calibration_data()
