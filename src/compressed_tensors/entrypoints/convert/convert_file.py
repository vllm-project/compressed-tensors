# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import COMPRESSION_VERSION_NAME, QUANTIZATION_CONFIG_NAME
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.utils.safetensors_load import (
    InverseWeightMap,
    find_config_path,
    load_tensors_from_inverse_weight_map,
)
from loguru import logger
from safetensors.torch import save_file


__all__ = [
    "validate_file",
    "convert_file",
    "write_checkpoint_quantization_config",
]


def write_checkpoint_quantization_config(
    save_directory: str | os.PathLike,
    converters: list[Converter],
    quant_config_data: dict,
):
    """
    Write the quantization config produced by `converter` into the model config
    file (config.json or params.json) in save_directory. This is called after
    the convert checkpoint pathway completes to record which quantization was
    applied. The quantization_config section is replaced entirely with the new
    config.

    :param save_directory: directory containing the model config file
    :param converter: Converter instance whose update_config() produces the
        updated quantization config
    """
    # apply updates to quantization config
    for converter in converters:
        quant_config_data = converter.update_config(quant_config_data)

    config_file_path = find_config_path(save_directory)
    if config_file_path is not None:
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        # add/remove quantization config data to config.json
        if not quant_config_data:
            if QUANTIZATION_CONFIG_NAME in config_data:
                del config_data[QUANTIZATION_CONFIG_NAME]
        else:
            quant_config_data[COMPRESSION_VERSION_NAME] = ct_version
            config_data[QUANTIZATION_CONFIG_NAME] = quant_config_data

        # write config.json
        with open(config_file_path, "w") as file:
            json.dump(config_data, file, indent=2, sort_keys=True)

    else:
        logger.warning(
            f"Could not find config file in {save_directory}. Please add to config "
            f"{json.dumps(quant_config_data, indent=2, sort_keys=True)}"
        )


def validate_file(
    inverse_weight_map: InverseWeightMap,
    converters: list[Converter],
):
    """
    Validate that each quantizable tensor in a safetensors file can be quantized.

    :param inverse_weight_map: mapping of resolved source file path ->
        list of tensor names to load from that file. Precomputed by
        build_inverse_weight_map() in the job-building phase.
        Example: {"/path/shard0.safetensors": ["q_proj.weight"],
                  "/path/shard1.safetensors": ["k_proj.weight", "v_proj.weight"]}
    :param converters: list of converters to apply to the checkpoint in sequence
    """
    tensors = load_tensors_from_inverse_weight_map(inverse_weight_map)

    for converter in converters:
        converter.validate(tensors)


def convert_file(
    inverse_weight_map: InverseWeightMap,
    save_path: str | os.PathLike,
    converters: list[Converter],
) -> tuple[int, dict[str, str]]:
    """
    Convert tensors in a given safetensors file

    :param inverse_weight_map: mapping of resolved source file path ->
        list of tensor names to load from that file. Precomputed by
        build_inverse_weight_map() in the job-building phase.
        Example: {"/path/shard0.safetensors": ["q_proj.weight"],
                  "/path/shard1.safetensors": ["k_proj.weight", "v_proj.weight"]}
    :param save_path: save path of file with quantized weights
    :param converters: list of converters to apply to the checkpoint in sequence
    :returns: tuple of (total_size, weight_map), respectively the total size in bytes
        of the saved file and dictionary of weight name -> save path
    """
    tensors = load_tensors_from_inverse_weight_map(inverse_weight_map)

    for converter in converters:
        converter.process(tensors)

    save_file(tensors, save_path)
    total_size = sum(tensor.nbytes for tensor in tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
    return total_size, weight_map
