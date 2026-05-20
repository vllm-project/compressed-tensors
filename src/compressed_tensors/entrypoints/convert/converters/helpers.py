# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from loguru import logger


__all__ = ["merge_quantization_config"]


def merge_quantization_config(
    config: dict[str, Any],
    new_config_groups: Optional[dict[str, QuantizationScheme]] = None,
    new_ignore: Optional[list[str]] = None,
    new_kv_cache_scheme: Optional[QuantizationArgs] = None,
    new_format: Optional[str] = None,
) -> dict[str, Any]:
    """
    Merge a new quantization config with an existing compressed-tensors config.

    :param config: The full model config containing quantization_config
    :param new_config_groups: New config groups to merge (update/append)
    :param new_ignore: New ignore list to concatenate with existing
    :param new_kv_cache_scheme: New kv_cache_scheme (errors if different from existing)
    :param new_format: New format (uses mixed_precision if different from existing)
    :return: Updated config dict
    """
    if not config:
        config = QuantizationConfig(config_groups={}).model_dump()

    if config.get("quant_method") != "compressed-tensors":
        logger.warning(
            "Cannot merge with existing quantization config with method "
            f"{config['quant_method']}. Manaually overwriting config, this may "
            "produce incorrect quantization configs"
        )
        config = QuantizationConfig(config_groups={}).model_dump()

    # Merge config_groups (update/append)
    existing_config_groups = config.get("config_groups", {})
    merged_config_groups = {**existing_config_groups}
    if new_config_groups:
        merged_config_groups.update(new_config_groups.model_dump())

    # Merge ignore lists (concatenate)
    existing_ignore = config.get("ignore", []) or []
    merged_ignore = list(existing_ignore)
    if new_ignore:
        merged_ignore.extend(new_ignore)

    # Handle kv_cache_scheme (error if not identical)
    existing_kv_cache = config.get("kv_cache_scheme")
    if existing_kv_cache is not None and new_kv_cache_scheme is not None:
        if existing_kv_cache != new_kv_cache_scheme:
            raise ValueError(
                f"Cannot merge configs with different kv_cache_schemes: "
                f"{existing_kv_cache} != {new_kv_cache_scheme}"
            )
    merged_kv_cache = existing_kv_cache or new_kv_cache_scheme

    # Handle format (use mixed precision if not identical)
    existing_format = config.get("format", new_format)
    merged_format = new_format or existing_format
    if existing_format != merged_format:
        merged_format = CompressionFormat.mixed_precision.value

    # Update quantization config with merged values
    config["config_groups"] = merged_config_groups
    config["ignore"] = merged_ignore
    config["kv_cache_scheme"] = merged_kv_cache
    config["format"] = merged_format
    # Always set quantization_status to compressed
    config["quantization_status"] = QuantizationStatus.COMPRESSED.value
    return config
