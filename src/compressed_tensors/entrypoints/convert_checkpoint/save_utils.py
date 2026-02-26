# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Iterable

from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import COMPRESSION_VERSION_NAME, QUANTIZATION_CONFIG_NAME
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert_checkpoint.converters import Converter
from compressed_tensors.quantization import QuantizationConfig, QuantizationStatus
from loguru import logger
from transformers.file_utils import CONFIG_NAME


__all__ = ["update_config", "update_safetensors_index"]


def update_config(
    save_directory: str | os.PathLike,
    converters: Iterable[Converter],
):
    schemes = [converter.create_scheme() for converter in converters]
    unique_formats = set(scheme.format for scheme in schemes)

    # construct quantization config
    quant_config = QuantizationConfig.model_validate(
        {
            "config_groups": {
                f"group_{group_idx}": scheme
                for (group_idx, scheme) in enumerate(schemes)
            },
            "quantization_status": QuantizationStatus.COMPRESSED,
            "format": (
                next(iter(unique_formats))
                if len(unique_formats) == 1
                else CompressionFormat.mixed_precision.value
            ),
        }
    )

    quant_config_data = quant_config.model_dump()
    quant_config_data[COMPRESSION_VERSION_NAME] = ct_version

    # write results to config.json file
    config_file_path = find_file_path(save_directory, (CONFIG_NAME, "params.json"))
    if config_file_path is not None:
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        config_data[QUANTIZATION_CONFIG_NAME] = quant_config_data

        with open(config_file_path, "w") as file:
            json.dump(config_data, file, indent=2, sort_keys=True)

    else:
        logger.warning(
            f"Could not find config file in {save_directory}. "
            f"Please {json.dumps(quant_config_data, indent=2, sort_keys=True)}"
        )


def update_safetensors_index(
    save_directory: str | os.PathLike,
    total_size: int,
    weight_map: dict[str, str],
):
    file_path = find_file_path(save_directory, "safetensors.index.json")
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


def find_file_path(
    save_directory: str | os.PathLike, file_names: str | list[str]
) -> str | None:
    if isinstance(file_names, str):
        file_names = [file_names]
    for file_name in os.listdir(save_directory):
        if file_name in file_names:
            return os.path.join(save_directory, file_name)

    return None
