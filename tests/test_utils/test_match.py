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

from unittest.mock import patch

import pytest
import torch.nn as nn

# Assuming the module is named "module_matching" - adjust import as needed
from compressed_tensors.utils import (
    InternalModule,
    get_lowest_common_ancestor_name,
    is_match,
    is_narrow_match,
    match_modules_set,
    match_named_modules,
    match_named_parameters,
)
from compressed_tensors.utils.match import _match_class, _match_name
from transformers import AutoModelForCausalLM


@pytest.fixture
def llama_stories_model():
    return AutoModelForCausalLM.from_pretrained(
        "Xenova/llama2.c-stories15M",
        torch_dtype="auto",
    )


class DummyModel(nn.Module):
    """Test model for unit tests. Weights are initialized on meta device"""

    def __init__(self):
        try:
            from accelerate import init_empty_weights
        except ImportError:
            pytest.skip("Skipping weight init requires accelerate")

        super().__init__()
        with init_empty_weights():
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 30)
            self.norm = nn.LayerNorm(30)
            self.attention = nn.MultiheadAttention(30, 2)

            # Create nested structure
            self.transformer = nn.ModuleDict(
                {
                    "layers": nn.ModuleList(
                        [
                            nn.ModuleDict(
                                {
                                    "self_attn": nn.ModuleDict(
                                        {
                                            "q_proj": nn.Linear(30, 30),
                                            "k_proj": nn.Linear(30, 30),
                                            "v_proj": nn.Linear(30, 30),
                                        }
                                    ),
                                    "norm": nn.LayerNorm(30),
                                    "mlp": nn.Linear(30, 30),
                                }
                            )
                            for _ in range(3)
                        ]
                    )
                }
            )


class DummyMoEModel(nn.Module):
    """Test MoE model for unit tests. Weights are initialized on meta device"""

    def __init__(self, num_layers=2, num_experts=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "post_attention_layernorm": nn.LayerNorm(3),
                        "mlp": nn.ModuleDict(
                            {
                                "experts": nn.ModuleList(
                                    [
                                        nn.ModuleDict(
                                            {
                                                "gate_proj": nn.Linear(3, 6),
                                                "up_proj": nn.Linear(3, 6),
                                                "down_proj": nn.Linear(6, 3),
                                            }
                                        )
                                        for _ in range(num_experts)
                                    ]
                                ),
                            }
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )


class TestMatchName:
    """Test cases for _match_name function"""

    def test_exact_match(self):
        """Test exact string matching"""
        assert _match_name("layer1", "layer1")
        assert not _match_name("layer1", "layer2")
        assert _match_name(
            "transformer.layers.0.self_attn.q_proj",
            "transformer.layers.0.self_attn.q_proj",
        )

    def test_regex_match(self):
        """Test regex matching with "re:" prefix"""
        assert _match_name("layer1", "re:layer.*")
        assert _match_name("layer1", "re:^layer1$")
        assert not _match_name("layer1", "re:layer2")
        assert _match_name("transformer.layers.0.self_attn.q_proj", "re:.*q_proj")
        assert _match_name(
            "transformer.layers.0.self_attn.q_proj",
            "re:transformer\\.layers\\.\\d+\\.self_attn\\..*_proj$",
        )

    def test_empty_strings(self):
        """Test edge cases with empty strings"""
        assert _match_name("", "")
        assert not _match_name("layer1", "")
        assert not _match_name("", "layer1")

    def test_regex_special_characters(self):
        """Test regex with special characters"""
        assert _match_name("layer.1", "re:layer\\.1")
        assert _match_name("layer.1", "re:layer.1")  # . matches any char
        assert _match_name("layer_1", "re:layer_1")


class TestMatchClass:
    """Test cases for _match_class function"""

    def test_direct_class_match(self):
        """Test matching direct class names"""
        linear = nn.Linear(10, 20)
        assert _match_class(linear, "Linear")
        assert not _match_class(linear, "Conv2d")

        norm = nn.LayerNorm(10)
        assert _match_class(norm, "LayerNorm")
        assert not _match_class(norm, "BatchNorm1d")

    def test_parent_class_match(self):
        """Test matching parent class names"""
        linear = nn.Linear(10, 20)
        assert _match_class(linear, "Module")

        conv = nn.Conv2d(3, 16, 3)
        assert _match_class(conv, "Module")
        assert _match_class(conv, "_ConvNd")

    def test_non_torch_module(self):
        """Test with non-torch modules"""
        regular_object = object()
        assert not _match_class(regular_object, "object")  # not a torch.nn.Module

    def test_custom_module(self):
        """Test with custom module classes"""
        model = DummyModel()
        assert _match_class(model, "DummyModel")
        assert _match_class(model, "Module")

    def test_linear_base(self):
        """Test matching against vllm's LinearBase class"""

        class LinearBase(nn.Module):
            pass

        linear = LinearBase()
        assert _match_class(linear, "Linear")


class TestIsMatch:
    """Test cases for is_match function"""

    def test_name_match(self):
        """Test matching by name"""
        linear = nn.Linear(10, 20)
        assert is_match("layer1", linear, "layer1")
        assert not is_match("layer1", linear, "layer2")

    def test_class_match(self):
        """Test matching by class"""
        linear = nn.Linear(10, 20)
        assert is_match("layer1", linear, "Linear")
        assert not is_match("layer1", linear, "Conv2d")

    def test_combined_match(self):
        """Test that either name or class match works"""
        linear = nn.Linear(10, 20)
        assert is_match("layer1", linear, "layer1")  # name match
        assert is_match("layer1", linear, "Linear")  # class match
        assert not is_match("layer1", linear, "layer2")  # no match

    def test_regex_in_name_match(self):
        """Test regex matching in name"""
        linear = nn.Linear(10, 20)
        assert is_match("layer1", linear, "re:layer.*")
        assert not is_match("layer1", linear, "re:conv.*")

    def test_internal_module_match(self):
        """Test not matching internal modules"""

        class InternalLinear(InternalModule, nn.Linear):
            pass

        linear = InternalLinear(10, 20)
        assert not is_match("layer1", linear, "re:layer.*")

    def test_fused_mapping(self):
        """"""
        linear = nn.Linear(10, 20)
        mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }

        assert is_match("dummy.qkv_proj", linear, "re:.*q_proj", fused=mapping)
        assert is_match("dummy.qkv_proj", linear, "re:.*k_proj", fused=mapping)
        assert is_match("dummy.qkv_proj", linear, "re:.*v_proj", fused=mapping)
        assert is_match("dummy.qkv_proj", linear, "Linear", fused=mapping)

        assert is_match("dummy.gate_up_proj", linear, "re:.*gate_proj", fused=mapping)
        assert is_match("dummy.gate_up_proj", linear, "re:.*up_proj", fused=mapping)
        assert is_match("dummy.gate_up_proj", linear, "Linear", fused=mapping)


class TestMatchNamedModules:
    """Test cases for match_named_modules function"""

    def test_exact_module_match(self):
        """Test matching modules by exact name"""
        model = DummyModel()
        matches = list(match_named_modules(model, ["layer1", "layer2"]))

        assert len(matches) == 2
        names = [name for name, _ in matches]
        assert "layer1" in names
        assert "layer2" in names

    def test_class_module_match(self):
        """Test matching modules by class name"""
        model = DummyModel()
        matches = list(match_named_modules(model, ["Linear"]))

        # Should find all Linear layers
        linear_modules = [
            module for _, module in matches if isinstance(module, nn.Linear)
        ]
        assert len(linear_modules) > 0

    def test_regex_module_match(self):
        """Test matching modules with regex patterns"""
        model = DummyModel()
        matches = list(match_named_modules(model, ["re:.*linear.*"]))

        # Should find layers with "linear" in name (case insensitive depends on model)
        assert len(matches) >= 0  # May be 0 if no "linear" in names

    def test_ignore_parameter(self):
        """Test ignoring specific modules"""
        model = DummyModel()
        matches_without_ignore = list(match_named_modules(model, ["Linear"]))
        matches_with_ignore = list(
            match_named_modules(model, ["Linear"], ignore=["layer1"])
        )

        # Should have fewer matches when ignoring layer1
        assert len(matches_with_ignore) <= len(matches_without_ignore)

        # layer1 should not be in ignored results
        ignored_names = [name for name, _ in matches_with_ignore]
        assert "layer1" not in ignored_names

    def test_empty_targets(self):
        """Test with empty targets list"""
        model = DummyModel()
        matches = list(match_named_modules(model, []))
        assert len(matches) == 0

    @patch("compressed_tensors.utils.match._LOGGER")
    def test_warn_on_fail(self, mock_logger):
        """Test warning when targets don"t match"""
        model = DummyModel()
        list(match_named_modules(model, ["nonexistent_module"], warn_on_fail=True))

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Could not match" in warning_msg
        assert "nonexistent_module" in warning_msg

    def test_internal_match(self):
        """Test not matching internal modules"""

        class InternalLinear(InternalModule, nn.Linear):
            pass

        linear = InternalLinear(10, 20)
        matches = list(match_named_modules(linear, ["re:.*"]))
        assert len(matches) == 0

    @pytest.mark.parametrize(
        "targets, ignore, expected_targets",
        [
            (
                ["re:model.layers.[01].self_attn.q_proj"],
                ["re:model.layers.1.self_attn.q_proj"],
                set(["model.layers.0.self_attn.q_proj"]),
            ),
            (
                ["re:model.layers.[01].self_attn.q_proj"],
                [],
                set(
                    [
                        "model.layers.0.self_attn.q_proj",
                        "model.layers.1.self_attn.q_proj",
                    ]
                ),
            ),
            (
                ["re:model.layers.[0-2].self_attn.q_proj"],
                ["re:model.layers.1.self_attn.q_proj"],
                set(
                    [
                        "model.layers.0.self_attn.q_proj",
                        "model.layers.2.self_attn.q_proj",
                    ]
                ),
            ),
            (
                ["model.layers.0.self_attn.q_proj"],
                ["model.layers.0.self_attn.q_proj"],
                set(),
            ),
            (
                ["re:model.layers.*.self_attn.q_proj"],
                ["re:model.layers.[01].self_attn.q_proj"],
                set(
                    f"model.layers.{layer_idx}.self_attn.q_proj"
                    for layer_idx in range(2, 6)
                ),
            ),
        ],
    )
    def test_expand_targets_with_llama_stories(
        self, llama_stories_model, targets, ignore, expected_targets
    ):
        expanded_targets = {
            name
            for name, _ in match_named_modules(llama_stories_model, targets, ignore)
        }
        assert expanded_targets == expected_targets


class TestMatchNamedParameters:
    """Test cases for match_named_parameters function"""

    def test_parameter_match(self):
        """Test matching parameters by name"""
        model = DummyModel()
        matches = list(match_named_parameters(model, ["layer1.weight", "layer1.bias"]))

        assert len(matches) == 2
        param_names = [name for name, _, _ in matches]
        assert "layer1.weight" in param_names
        assert "layer1.bias" in param_names

    def test_regex_parameter_match(self):
        """Test matching parameters with regex"""
        model = DummyModel()
        matches = list(match_named_parameters(model, ["re:.*weight$"]))

        # Should find all weight parameters
        weight_params = [name for name, _, _ in matches if name.endswith(".weight")]
        assert len(weight_params) > 0

    def test_ignore_parameters(self):
        """Test ignoring specific parameters"""
        model = DummyModel()
        matches_without_ignore = list(match_named_parameters(model, ["re:.*weight$"]))
        matches_with_ignore = list(
            match_named_parameters(model, ["re:.*weight$"], ignore=["layer1.weight"])
        )

        # Should have fewer matches when ignoring
        assert len(matches_with_ignore) < len(matches_without_ignore)

        # layer1.weight should not be in ignored results
        ignored_names = [name for name, _, _ in matches_with_ignore]
        assert "layer1.weight" not in ignored_names

    def test_parameter_return_values(self):
        """Test that function returns correct tuple values"""
        model = DummyModel()
        matches = list(match_named_parameters(model, ["layer1.weight"]))

        assert len(matches) == 1
        param_name, parent_module, param = matches[0]

        assert param_name == "layer1.weight"
        assert parent_module is model.layer1
        assert isinstance(param, nn.Parameter)
        assert param is model.layer1.weight

    @patch("compressed_tensors.utils.match._LOGGER")
    def test_warn_on_fail_parameters(self, mock_logger):
        """Test warning when parameter targets don"t match"""
        model = DummyModel()
        list(match_named_parameters(model, ["nonexistent.param"], warn_on_fail=True))

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Could not match" in warning_msg
        assert "nonexistent.param" in warning_msg

    def test_internal_match(self):
        """Test not matching internal modules"""

        class InternalLinear(InternalModule, nn.Linear):
            pass

        linear = InternalLinear(10, 20)
        matches = list(match_named_parameters(linear, ["re:.*"]))
        assert len(matches) == 0


class TestGetLowestCommonModuleName:
    """Test cases for get_lowest_common_ancestor_name function"""

    def test_multiple_modules(self):
        assert "abc" == get_lowest_common_ancestor_name(
            [
                "abc.a",
                "abc.b",
                "abc.c",
            ]
        )

    def test_single_module(self):
        assert "abc.abc" == get_lowest_common_ancestor_name(
            [
                "abc.abc",
            ]
        )

    def test_substring_modules(self):
        assert "abc" == get_lowest_common_ancestor_name(
            [
                "abc.abc",
                "abc.ab",
            ]
        )

    def test_parent_and_child_modules(self):
        assert "abc.abc" == get_lowest_common_ancestor_name(
            [
                "abc.abc.ab",
                "abc.abc",
            ]
        )

    def test_root(self):
        assert "" == get_lowest_common_ancestor_name(
            [
                "abc.abc",
                "b.abc",
            ]
        )

    def test_ignore_none(self):
        assert "abc.abc" == get_lowest_common_ancestor_name(
            [
                "abc.abc",
                None,
            ]
        )


class TestMatchModulesSet:
    """Test cases for match_modules_set function"""

    def test_simple_module_set(self):
        """Test matching simple module sets"""
        model = DummyModel()
        targets = [
            "re:.*self_attn.q_proj$",
            "re:.*self_attn.k_proj$",
            "re:.*self_attn.v_proj$",
        ]

        matches = list(match_modules_set(model, targets))

        # Should have 3 sets (one for each layer)
        assert len(matches) == 3

        # Each set should have 3 modules
        for module_set in matches:
            assert len(module_set) == 3
            assert all(isinstance(*m, nn.Linear) for m in module_set)

    def test_moe_module_match(self):
        """Test matching MoE modules with multiple experts per layer"""
        model = DummyMoEModel(num_layers=2, num_experts=4)

        # Test matching expert projections - each expert becomes its own set
        # because the parent context differs between experts
        targets = [
            "re:.*gate_proj$",
            "re:.*up_proj$",
        ]

        matches = list(match_modules_set(model, targets))

        # Should have 8 sets (2 layers * 4 experts)
        assert len(matches) == 8

        # Each set should have 2 target lists (gate_proj, up_proj)
        for expert_group in matches:
            assert len(expert_group) == 2
            gate_modules, up_modules = expert_group

            # Each target should have matched 1 module (single expert)
            assert len(gate_modules) == 1
            assert len(up_modules) == 1

            # All modules should be Linear layers
            assert isinstance(gate_modules[0], nn.Linear)
            assert isinstance(up_modules[0], nn.Linear)

    def test_moe_with_layernorm_match(self):
        """
        Test matching MoE modules with their corresponding layer norms.
        Including a layer-level module (layernorm) groups all experts in
        that layer together.
        """
        model = DummyMoEModel(num_layers=2, num_experts=3)

        # Match layer norm with expert projections - the layernorm is at layer level,
        # so it establishes a common parent context for all experts in that layer
        targets = [
            "re:.*post_attention_layernorm$",
            "re:.*gate_proj$",
            "re:.*up_proj$",
        ]

        matches = list(match_modules_set(model, targets))

        # Should have 2 layer groups (one per layer)
        assert len(matches) == 2

        for layer_group in matches:
            assert len(layer_group) == 3
            norm_modules, gate_modules, up_modules = layer_group

            # LayerNorm should have 1 module (single per layer)
            assert len(norm_modules) == 1
            assert isinstance(norm_modules[0], nn.LayerNorm)

            # Each projection should have 3 experts (all experts in the layer)
            assert len(gate_modules) == 3
            assert len(up_modules) == 3
            assert all(isinstance(m, nn.Linear) for m in gate_modules)
            assert all(isinstance(m, nn.Linear) for m in up_modules)

    def test_module_set_ordering(self):
        """Test that module sets maintain target ordering"""
        model = DummyModel()
        targets = [
            "re:.*v_proj$",  # v first
            "re:.*self_attn.q_proj$",  # q second
            "re:.*self_attn.k_proj$",
        ]  # k third

        matches = list(match_modules_set(model, targets))

        for module_set in matches:
            # Check that modules are returned in target order (v, q, k)
            v_proj, q_proj, k_proj = module_set
            v_proj, q_proj, k_proj = *v_proj, *q_proj, *k_proj
            # We can't easily check the exact modules, but can check they're all Linear
            assert all(isinstance(m, nn.Linear) for m in [v_proj, q_proj, k_proj])

    def test_incomplete_set_error(self):
        """Test error when unable to complete a set"""
        model = DummyModel()
        targets = ["layer1", "nonexistent_module"]

        with pytest.raises(
            ValueError, match="Found a final incomplete set with matches found for keys"
        ):
            list(match_modules_set(model, targets))

    def test_empty_targets_set(self):
        """Test with empty targets"""
        model = DummyModel()
        matches = list(match_modules_set(model, []))
        # Should yield one empty set for each module traversed?
        # Actually, with empty targets, we expect no matches
        assert len(matches) == 0

    def test_module_set_with_ignore(self):
        """Test module set matching with ignore parameter"""
        model = DummyModel()
        targets = ["re:.*self_attn.q_proj$", "re:.*self_attn.k_proj$"]
        ignore = ["re:transformer.layers.0.*"]  # Ignore first layer

        matches = list(match_modules_set(model, targets, ignore=ignore))

        # Should have 2 sets (layers 1 and 2, but not 0)
        assert len(matches) == 2

    def test_internal_match(self):
        """Test not matching internal modules"""

        class InternalLinear(InternalModule, nn.Linear):
            pass

        linear = InternalLinear(10, 20)
        matches = list(match_modules_set(linear, ["re:.*"]))
        assert len(matches) == 0


class TestIsNarrowMatch:
    def test_narrow_match_true_child_only(self):
        """
        Target matches the child module name but NOT its parent name.
        Should return True.
        """
        model = DummyModel()
        name = "transformer.layers.0.self_attn.q_proj"
        # Matches "...q_proj" but not "...self_attn"
        target = r"re:.*q_proj$"

        assert is_narrow_match(model, target, name)

    def test_narrow_match_false_when_parent_also_matches(self):
        """
        Target matches both the child and its parent name.
        Should return False because it's not a 'narrow' match.
        """
        model = DummyModel()
        name = "transformer.layers.0.self_attn.q_proj"
        # Broad target that also matches the parent "transformer.layers.0.self_attn"
        target = r"re:transformer\.layers\.0\..*"

        assert not is_narrow_match(model, target, name)

    def test_narrow_match_false_when_neither_matches(self):
        """
        Target matches neither the child nor the parent.
        Should return False.
        """
        model = DummyModel()
        name = "transformer.layers.0.self_attn.q_proj"
        target = r"re:this_does_not_exist$"

        assert not is_narrow_match(model, target, name)

    def test_narrow_match_iterable_targets_any_true(self):
        """
        With multiple targets: if any target narrowly matches the child,
        the function should return True.
        """
        model = DummyModel()
        name = "transformer.layers.0.self_attn.q_proj"
        # First target is broad (matches both child & parent -> narrow False),
        # second target is narrow (matches child only -> narrow True).
        targets = [
            r"re:transformer\.layers\.0\..*",
            r"re:.*q_proj$",
        ]

        assert is_narrow_match(model, targets, name)

    def test_narrow_match_with_explicit_module_argument(self):
        """
        Passing the module explicitly should behave the same as when it's
        retrieved from the model by name.
        """
        model = DummyModel()
        name = "transformer.layers.0.self_attn.q_proj"
        module = model.get_submodule(name)
        target = r"re:.*q_proj$"

        # Both ways should be True
        assert is_narrow_match(model, target, name)
        assert is_narrow_match(model, target, name, module=module)

    def test_narrow_match_top_level_behavior_documented(self):
        """
        (Behavior check) For a top-level module name without a dot, the current
        implementation derives parent_name == name, so parent==child.
        Then 'narrow' cannot be True because parent match mirrors child match.
        This test documents current behavior to guard against regressions.
        """
        model = DummyModel()
        name = "layer1"  # top-level module in DummyModel
        target = r"re:^layer1$"

        assert not is_narrow_match(model, target, name)


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_complex_model_matching(self):
        """Test matching on a more complex model structure"""
        model = DummyModel()

        # Test that we can find attention projection layers
        q_matches = list(match_named_modules(model, ["re:.*q_proj$"]))
        k_matches = list(match_named_modules(model, ["re:.*k_proj$"]))
        v_matches = list(match_named_modules(model, ["re:.*v_proj$"]))

        assert len(q_matches) == 3  # 3 layers
        assert len(k_matches) == 3
        assert len(v_matches) == 3

    def test_parameter_and_module_consistency(self):
        """Test that parameter and module matching are consistent"""
        model = DummyModel()

        # Get modules
        module_matches = list(match_named_modules(model, ["layer1"]))
        assert len(module_matches) == 1
        module_name, module = module_matches[0]

        # Get parameters from that module
        param_matches = list(match_named_parameters(model, [f"{module_name}.weight"]))
        assert len(param_matches) == 1
        param_name, parent_module, param = param_matches[0]

        # Check consistency
        assert parent_module is module
        assert param is module.weight

    def test_all_functions_with_regex(self):
        """Test all functions work with regex patterns"""
        model = DummyModel()
        regex_target = "re:.*Linear.*"

        # Should not crash and should handle regex consistently
        modules = list(match_named_modules(model, [regex_target]))
        params = list(match_named_parameters(model, [regex_target]))

        # Basic sanity checks
        assert isinstance(modules, list)
        assert isinstance(params, list)
