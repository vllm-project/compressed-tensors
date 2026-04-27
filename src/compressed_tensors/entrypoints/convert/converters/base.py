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
        raise NotImplementedError()

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Validation layer to quickly log warnings or raise an error if the safetensors
        file is not compatible with Converter.

        :param tensors: dictionary of tensor name to tensor, as loaded from
        safetensors file.
        """
        raise NotImplementedError()

    def create_config(self, config: dict) -> dict:
        """
        Update quantization config to reflect processing done by this converter

        :param config: existing quantization config
        :return: new quantization config with processing
        """
        raise NotImplementedError()

    def get_dependencies(self, weight_name: str) -> set[str]:
        """
        Given a weight name, return a set of all dependency weight names, so that
        weights can be processed correctly and in a parallelized fashion.
        If there are no dependencies, an empty dict should be returned.

        :returns: set[str] of dependency weight names
        """
        raise NotImplementedError()


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

    def get_dependencies_recursive(
        weight_name: str, converters: list[Converter], current_deps: set[str]
    ) -> set[str]:
        for converter in converters:
            deps = converter.get_dependencies(weight_name)
            for dep in deps:
                if dep not in current_deps:
                    current_deps.add(dep)
                    get_dependencies_recursive(dep, converters, current_deps)

        return current_deps

    # map of weight name -> set of dependency names
    weight_deps_dict: dict[str, set[str]] = dict()
    for weight_name in weight_map:
        weight_deps_dict[weight_name] = get_dependencies_recursive(
            weight_name, converters, set()
        )
        assert (
            weight_name not in weight_deps_dict[weight_name]
        ), f"{weight_name} found in dependencies {weight_deps_dict[weight_name]}"

    # set of all dependencies (i.e. all weight names required by another)
    all_dependencies: set[str] = set().union(*weight_deps_dict.values())

    inverse_weight_maps: dict[str, InverseWeightMap] = defaultdict(
        lambda: defaultdict(list)
    )
    for weight_name, weight_shard_name in weight_map.items():
        if weight_name in all_dependencies:
            # weight is a partner to some other primary tensor, skip it
            continue

        # weight is purely a primary weight, is not a dependency of anything
        # add it and all its dependencies
        current_iwm: InverseWeightMap = inverse_weight_maps[weight_shard_name]
        dependency_weights = weight_deps_dict[weight_name]
        for weight_to_add_name in [
            weight_name,
            *dependency_weights,
        ]:
            if weight_to_add_name not in weight_map:
                raise ValueError(
                    f"Dependency weight {weight_to_add_name} not found in weight map"
                )
            weight_to_add_shard_name = weight_map[weight_to_add_name]
            resolved_path = model_files[weight_to_add_shard_name]
            current_iwm[resolved_path].append(weight_to_add_name)

    # return dicts, not defaultdicts, to avoid silent errors
    return {k: dict(v) for k, v in inverse_weight_maps.items()}
