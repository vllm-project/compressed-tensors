# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import COMPRESSION_VERSION_NAME, QUANTIZATION_CONFIG_NAME
from compressed_tensors.entrypoints.convert import (
    Converter,
    find_config_path,
    find_safetensors_index_path,
)
from loguru import logger
from transformers.file_utils import CONFIG_NAME


__all__ = ["update_config", "update_safetensors_index"]


def update_config(
    save_directory: str | os.PathLike,
    converter: Converter,
):
    """
    Update Quantization config for model stub in save_directory,
    based on the converter that was used.
    Quantization config is considered stale and re-written entirely.
    """
    quant_config = converter.create_config()

    quant_config_data = quant_config.model_dump()
    quant_config_data[COMPRESSION_VERSION_NAME] = ct_version

    # write results to config.json or params.json file
    config_file_path = find_config_path(save_directory)
    if config_file_path is not None:
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        config_data[QUANTIZATION_CONFIG_NAME] = quant_config_data

        with open(config_file_path, "w") as file:
            json.dump(config_data, file, indent=2, sort_keys=True)

    else:
        logger.warning(
            f"Could not find config file in {save_directory}. Please add to config "
            f"{json.dumps(quant_config_data, indent=2, sort_keys=True)}"
        )


def update_safetensors_index(
    save_directory: str | os.PathLike,
    total_size: int,
    weight_map: dict[str, str],
):
    file_path = find_safetensors_index_path(save_directory)
    if file_path is None:
        return

    with open(file_path, "w") as file:
        json.dump(
            {
                "metadata": {
                    "total_size": total_size,
                },
                "weight_map": weight_map,
            },
            file,
            indent=2,
            sort_keys=True,
        )
