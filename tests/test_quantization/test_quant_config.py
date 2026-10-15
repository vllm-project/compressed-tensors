# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_FORMAT,
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
)
from compressed_tensors.quantization.quant_config import (
    _map_to_checkpoint_names,
    get_vllm_module_type,
)
from compressed_tensors.utils.match import match_named_modules
from pydantic import ValidationError
from transformers import AutoModelForImageTextToText
from transformers.core_model_loading import WeightRenaming


def test_basic_config():
    config_groups = {"group_1": QuantizationScheme(targets=[])}
    config = QuantizationConfig(config_groups=config_groups)

    assert config.config_groups == config_groups
    assert config.quant_method == DEFAULT_QUANTIZATION_METHOD
    assert config.format == DEFAULT_QUANTIZATION_FORMAT
    assert config.quantization_status == QuantizationStatus.INITIALIZED
    assert config.global_compression_ratio is None
    assert isinstance(config.ignore, list) and len(config.ignore) == 0


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


@pytest.mark.parametrize("targets", [["runtime.0"], [r"re:^runtime\."], ["Linear"]])
def test_export_targets_uses_checkpoint_names_without_mutating_scheme(targets):
    model = torch.nn.Module()
    model.runtime = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model._weight_conversions = [
        WeightRenaming("checkpoint.projection.weight", "runtime.0.weight"),
        WeightRenaming("checkpoint.ignored.weight", "runtime.1.weight"),
    ]
    scheme = QuantizationScheme(targets=targets, weights=QuantizationArgs())
    model.runtime[0].quantization_scheme = scheme
    model.runtime[0].quantization_status = QuantizationStatus.FROZEN

    first = QuantizationConfig.from_pretrained(model)
    second = QuantizationConfig.from_pretrained(model)

    assert first == second
    assert first.config_groups["group_0"].targets == (
        ["Linear"] if targets == ["Linear"] else ["checkpoint.projection"]
    )
    assert first.ignore == ["checkpoint.ignored"]
    assert model.runtime[0].quantization_scheme.targets == targets


def test_export_mixed_targets_preserves_backbone_and_mtp_assignments():
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(4, 4)
    model.mtp = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model._weight_conversions = [
        WeightRenaming("checkpoint.backbone.weight", "backbone.weight"),
        WeightRenaming("checkpoint.mtp.weight", "mtp.0.weight"),
    ]
    targets = ["Linear", r"re:^mtp\."]
    scheme = QuantizationScheme(targets=targets, weights=QuantizationArgs())
    for module in (model.backbone, model.mtp[0]):
        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.FROZEN

    first = QuantizationConfig.from_pretrained(model)
    second = QuantizationConfig.from_pretrained(model)

    assert first == second
    assert first.config_groups["group_0"].targets == [
        "checkpoint.backbone",
        "checkpoint.mtp",
    ]
    assert scheme.targets == targets


@pytest.mark.parametrize("copy_scheme", [False, True])
def test_export_named_targets_preserves_overlapping_group_assignments(copy_scheme):
    model = torch.nn.Module()
    model.runtime = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(3)))
    model._weight_conversions = [WeightRenaming("checkpoint.", "runtime.")]
    broad = QuantizationScheme(targets=[r"re:^runtime\."], weights=QuantizationArgs())
    specific = QuantizationScheme(
        targets=["runtime.1"], weights=QuantizationArgs(num_bits=4)
    )
    model.runtime[0].quantization_scheme = broad
    model.runtime[1].quantization_scheme = specific
    model.runtime[2].quantization_scheme = broad.model_copy() if copy_scheme else broad
    for module in model.runtime:
        module.quantization_status = QuantizationStatus.FROZEN

    exported = QuantizationConfig.from_pretrained(model)

    assert exported.config_groups["group_0"].targets == ["checkpoint.0", "checkpoint.2"]
    assert exported.config_groups["group_1"].targets == ["checkpoint.1"]
    assert broad.targets == [r"re:^runtime\."]
    assert specific.targets == ["runtime.1"]


@pytest.mark.parametrize("include_linear", [False, True])
def test_export_preserves_kv_cache_assignments(include_linear):
    class CacheAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = torch.nn.Linear(4, 4)

        def forward(self, hidden_states, past_key_value=None):
            return hidden_states

    model = torch.nn.Module()
    model.runtime = torch.nn.ModuleDict({"attention": CacheAttention()})
    model._weight_conversions = [WeightRenaming("checkpoint.", "runtime.")]
    scheme = QuantizationScheme(
        targets=[r"re:^runtime\."], input_activations=QuantizationArgs()
    )
    model.runtime["attention"].quantization_scheme = scheme.model_copy()
    model.runtime["attention"].quantization_status = QuantizationStatus.FROZEN
    if include_linear:
        model.runtime["linear"] = torch.nn.Linear(4, 4)
        model.runtime["linear"].quantization_scheme = scheme
        model.runtime["linear"].quantization_status = QuantizationStatus.FROZEN

    exported = QuantizationConfig.from_pretrained(model)

    assert exported.kv_cache_scheme == scheme.input_activations
    if include_linear:
        assert exported.config_groups["group_0"].targets == [
            "checkpoint.attention",
            "checkpoint.linear",
        ]
    else:
        assert exported.config_groups == {}


def test_export_excludes_unquantized_module_renamed_into_target():
    model = torch.nn.Module()
    model.runtime = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model.other = torch.nn.Embedding(4, 4)
    model._weight_conversions = [WeightRenaming("runtime.saved.weight", "other.weight")]
    scheme = QuantizationScheme(targets=[r"re:^runtime\."], weights=QuantizationArgs())
    model.runtime[0].quantization_scheme = scheme
    model.runtime[0].quantization_status = QuantizationStatus.FROZEN

    exported = QuantizationConfig.from_pretrained(model)

    # Embedding is not in ignore: retaining the regex would quantize it on reload.
    assert exported.ignore == []
    assert exported.config_groups["group_0"].targets == ["runtime.0"]
    checkpoint = torch.nn.Module()
    checkpoint.runtime = torch.nn.ModuleDict(
        {"0": torch.nn.Linear(4, 4), "saved": torch.nn.Embedding(4, 4)}
    )
    apply_quantization_config(checkpoint, exported)
    assert checkpoint.runtime["0"].quantization_scheme.weights == scheme.weights
    assert not hasattr(checkpoint.runtime["saved"], "quantization_scheme")


@pytest.mark.parametrize(
    "targets",
    [["attention.0"], [r"re:^attention\."], ["Linear", r"re:^attention\."]],
)
def test_export_preserves_targets_with_unrelated_conversions(targets):
    model = torch.nn.Module()
    model.attention = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(64)))
    model.expert = torch.nn.Embedding(4, 4)
    model._weight_conversions = [
        WeightRenaming("checkpoint.expert.weight", "expert.weight")
    ]
    scheme = QuantizationScheme(targets=targets, weights=QuantizationArgs())
    for _, module in match_named_modules(model, targets):
        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.FROZEN

    first = QuantizationConfig.from_pretrained(model)
    second = QuantizationConfig.from_pretrained(model)

    assert first == second
    assert first.config_groups["group_0"].targets == targets
    assert scheme.targets == targets


def test_export_preserves_regex_when_renamed_modules_still_match():
    model = torch.nn.Module()
    model.attention = torch.nn.Module()
    model.attention.runtime = torch.nn.Linear(4, 4)
    model._weight_conversions = [
        WeightRenaming("attention.saved.weight", "attention.runtime.weight")
    ]
    targets = [r"re:^attention\."]
    scheme = QuantizationScheme(targets=targets, weights=QuantizationArgs())
    model.attention.runtime.quantization_scheme = scheme
    model.attention.runtime.quantization_status = QuantizationStatus.FROZEN

    exported = QuantizationConfig.from_pretrained(model)

    assert exported.config_groups["group_0"].targets == targets
    checkpoint = torch.nn.Module()
    checkpoint.attention = torch.nn.Module()
    checkpoint.attention.saved = torch.nn.Linear(4, 4)
    apply_quantization_config(checkpoint, exported)
    assert checkpoint.attention.saved.quantization_scheme.weights == scheme.weights


def test_export_preserves_groups_when_rename_enters_another_groups_regex():
    model = torch.nn.Module()
    model.runtime = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model.other = torch.nn.Linear(4, 4)
    model._weight_conversions = [WeightRenaming("runtime.1.weight", "other.weight")]
    broad = QuantizationScheme(targets=[r"re:^runtime\."], weights=QuantizationArgs())
    specific = QuantizationScheme(
        targets=["other"], weights=QuantizationArgs(num_bits=4)
    )
    model.runtime[0].quantization_scheme = broad
    model.other.quantization_scheme = specific
    for module in (model.runtime[0], model.other):
        module.quantization_status = QuantizationStatus.FROZEN

    exported = QuantizationConfig.from_pretrained(model)

    assert exported.config_groups["group_0"].targets == ["runtime.0"]
    assert exported.config_groups["group_1"].targets == ["runtime.1"]
    checkpoint = torch.nn.Module()
    checkpoint.runtime = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    )
    apply_quantization_config(checkpoint, exported)
    assert checkpoint.runtime[0].quantization_scheme.weights.num_bits == 8
    assert checkpoint.runtime[1].quantization_scheme.weights.num_bits == 4


def test_export_preserves_exact_target_priority_over_regex_after_rename():
    model = torch.nn.Module()
    model.runtime = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model._weight_conversions = [
        WeightRenaming("runtime.saved.weight", "runtime.0.weight")
    ]
    specific = QuantizationScheme(
        targets=["runtime.0", r"re:^runtime\."], weights=QuantizationArgs()
    )
    broad = QuantizationScheme(
        targets=[r"re:^runtime\."], weights=QuantizationArgs(num_bits=4)
    )
    model.runtime[0].quantization_scheme = specific
    model.runtime[1].quantization_scheme = broad
    for module in model.runtime:
        module.quantization_status = QuantizationStatus.FROZEN

    exported = QuantizationConfig.from_pretrained(model)

    assert exported.config_groups["group_0"].targets == ["runtime.saved"]
    assert exported.config_groups["group_1"].targets == broad.targets
    checkpoint = torch.nn.Module()
    checkpoint.runtime = torch.nn.ModuleDict(
        {"saved": torch.nn.Linear(4, 4), "1": torch.nn.Linear(4, 4)}
    )
    apply_quantization_config(checkpoint, exported)
    assert checkpoint.runtime["saved"].quantization_scheme.weights.num_bits == 8
    assert checkpoint.runtime["1"].quantization_scheme.weights.num_bits == 4


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
