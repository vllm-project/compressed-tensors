# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import COMPRESSION_VERSION_NAME, QUANTIZATION_CONFIG_NAME
from compressed_tensors.utils.safetensors_load import find_config_path
from loguru import logger


if TYPE_CHECKING:
    from compressed_tensors.entrypoints.convert.converters.base import Converter


__all__ = ["update_config"]


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
