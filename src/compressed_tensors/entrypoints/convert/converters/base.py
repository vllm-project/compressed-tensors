# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

import torch

__all__ = ["Converter", "build_inverse_weights_map"]

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
        to process weight_name correctly
        """
        pass

    def is_required_by(self, weight_name: str) -> set[str]:
        """
        Given a weight name, return the set of all weight names that require
        weight_name, in order for them to be processed correctly
        """
        pass


def build_inverse_weights_map(
    shard_name: str,
    weight_map: dict[str, str],
    model_files: dict[str, str],
    converters: list[Converter],
) -> dict[str, list[str]]:
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

    inverse_weights_map: dict[str, list[str]] = defaultdict(list)
    for weight_name, weight_shard_name in weight_map.items():
        if weight_shard_name != shard_name:
            continue

        if any([converter.is_required_by(weight_name) for converter in converters]):
            # weight is a partner to some other primary tensor, skip it
            continue

        requires = get_recursive_requires(weight_name, converters, {})
        if any(
            [
                converter.is_required_by(name)
                for name in requires
                for converter in converters
            ]
        ):
            # weight's required partner is required by another converter, skip it
            continue

        # if weight or its dependencies are not required by any tensor,
        # include it and all its requirements
        resolved_path = model_files.get(weight_shard_name)
        inverse_weights_map[resolved_path].append(weight_name)

        for require in requires:
            resolved_path = model_files.get(weight_map[require])
            inverse_weights_map[resolved_path].append(require)

    return dict(inverse_weights_map)
