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
    Converter interface for modifying safetensors checkpoints.

    Converters can be chained: the pipeline passes each file through a list of
    converters in order, feeding one converter's output to the next.
    """

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process tensors, returning converted set to be saved in a safetensors file.
        Converted tensors are dequantized (i.e. no quantization config) or in a
        compressed-tensors compatible format.

        Examples:
        - rename tensor or invert weights to match compressed-tensors convention.
        - dequantize to full-precision

        :param tensors: dictionary of tensor name to tensor, as loaded from
        safetensors file. Tensor name is a concatenation of module name and
        parameter name, e.g.
        - `model.layers.0.self_attn.q_proj.weight`
        - `model.layers.0.mlp.up_proj.weight_packed`

        :returns: dictionary of converted tensor name to tensor, to be saved in a
        safetensors file. Same format as input param tensors.
        """
        raise NotImplementedError()

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Validation layer to quickly log warnings or raise an error if the
        safetensors file is not compatible with the Converter. Returns the
        tensors that `process` would produce, so that chained converters can
        see the output format (tensor names and dtypes) of the one before them.

        By default this simply calls `process`, which is meta-safe for most converters
        (renames, dtype casts, inversions). Converters whose `process` cannot run
        on meta tensors (e.g. AutoAWQ) should override this
        to simulate the output instead.

        :param tensors: dictionary of tensor name to tensor, as loaded from
        safetensors file.
        :returns: dictionary of converted tensor name to tensor, matching what
        `process` would produce.
        """
        return self.process(tensors)

    def update_config(
        self,
        config: QuantizationConfig | None,
        save_directory: str | None = None,
    ) -> QuantizationConfig | None:
        """
        Build or update the QuantizationConfig for config.json and optionally
        apply any other changes to the model config in save_directory.

        When converters are chained, each receives the previous converter's
        config output. Re-quantizers merge their config into the existing one;
        dequantizers return None to strip quantization_config entirely.
        Converters that need to update non-quantization fields in config.json
        (e.g. expert count after pruning) should do so via save_directory.

        :param config: config from the previous converter, or None if this
            is the first converter (or if a previous dequantizer cleared it)
        :param save_directory: output directory containing config.json, or
            None if not yet available (e.g. during validation)
        :returns: updated QuantizationConfig, or None to remove it
        """
        raise NotImplementedError()

    def update_model_config(self, model_config: dict) -> dict:
        """
        Update non-quantization fields of the model config (config.json) dict.

        Most converters only touch the quantization_config (see `update_config`),
        so this is a pass-through by default. Converters that alter model
        structure (e.g. pruning experts) override this to mutate the config dict,
        while the pipeline owns loading and saving the file.

        :param model_config: the parsed config.json dict
        :returns: the updated config dict
        """
        return model_config

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
