# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

import torch
from compressed_tensors.utils.safetensors_load import InverseWeightMap


__all__ = ["Converter", "build_inverse_weight_maps"]

if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationConfig


class Converter(Protocol):
    """
    Converter interface, to modify safetensors files based on tensor name and
    pointer to torch.Tensor, and create the QuantizationConfig
    """

    def process(self, tensors: dict[str, torch.Tensor]):
        """
        Operate on safetensors file in-place, to convert it into a compressed-tensors
        compatible format.
        e.g. rename tensor, or invert weights to match compressed-tensors convention.

        :param tensors: dictionary of tensor name to tensor, as loaded from
        safetensors file. Tensor name is a concatenation of module name and
        parameter name, e.g.
        - `model.layers.0.self_attn.q_proj.weight`
        - `model.layers.0.mlp.up_proj.weight_packed`
        """
        pass

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Validation layer to quickly log warnings or raise an error if the safetensors
        file is not compatible with Converter.

        :param tensors: dictionary of tensor name to tensor, as loaded from
        safetensors file.
        """
        pass

    def create_config(self) -> QuantizationConfig | None:
        """
        Create compressed-tensors QuantizationConfig so that it can be set in the
        new model checkpoint's config.json.
        If the converter is moving checkpoint to full-precision, have this function
        return None, and quantization_config will be removed from config.json
        """
        pass

    def requires(self, weight_name: str) -> set[str]:
        """
        Given a weight name, return the set of all weight names required in order
        to process weight_name correctly.
        If there is no dependency, an empty set is returned.
        """
        pass


def build_inverse_weight_maps(
    weight_map: dict[str, str],
    model_files: dict[str, str],
    converters: list[Converter],
) -> dict[str, InverseWeightMap]:
    """
    For a given output shard, precompute exactly which tensors to load from
    which source files — including required partner tensors from other shards.

    This is necessary because some converters require that a set of tensors are
    accessible in order for them to be processed correctly.

    :param shard_name: the shard filename this job will process and save
    :param weight_map: tensor name -> shard filename (from safetensors.index.json)
    :param model_files: shard filename -> resolved absolute path
    :return: {resolved_file_path: [tensor_names_to_load]}
    """

    def get_recursive_requires(
        weight_name: str, converters: list[Converter], current_requires: set[str]
    ) -> tuple[set[str], set[str]]:
        for converter in converters:
            for require in converter.requires(weight_name):
                if require not in current_requires:
                    current_requires.add(require)
                    get_recursive_requires(require, converters, current_requires)

        return current_requires

    # map of weight name -> set of weights required to process this weight
    weight_requires_dict: dict[str, set[str]] = defaultdict(set)
    for weight_name, weight_shard_name in weight_map.items():
        weight_requires_dict[weight_name] = get_recursive_requires(
            weight_name, converters, set()
        )
        assert (
            weight_name not in weight_requires_dict[weight_name]
        ), f"{weight_name} found in requires {weight_requires_dict[weight_name]}"

    # set of all weights that are dependencies (i.e. required by a primary weight)
    dependency_weights: set[str] = set()
    for values in weight_requires_dict.values():
        for value in values:
            dependency_weights.add(value)

    inverse_weight_maps: dict[str, InverseWeightMap] = defaultdict(
        lambda: defaultdict(list)
    )
    for weight_name, weight_shard_name in weight_map.items():
        if weight_name in dependency_weights:
            # weight is a partner to some other primary tensor, skip it
            continue

        # weight is purely a primary weight, is not a dependency of anything
        # add it and all its required weights
        inverse_weight_map: InverseWeightMap = inverse_weight_maps[weight_shard_name]
        required_weights = weight_requires_dict[weight_name]
        for weight_to_add_name in [weight_name, *required_weights]:
            weight_to_add_shard_name = weight_map[weight_to_add_name]
            resolved_path = model_files.get(weight_to_add_shard_name)
            inverse_weight_map[resolved_path].append(weight_to_add_name)

    # return dicts, not defaultdicts
    return {k: dict(v) for k, v in inverse_weight_maps.items()}
