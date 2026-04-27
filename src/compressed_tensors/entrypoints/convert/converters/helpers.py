# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationStatus


__all__ = ["merge_quantization_config"]


def merge_quantization_config(
    config: dict[str, Any],
    new_config_groups: dict[str, Any] | None = None,
    new_ignore: list[str] | None = None,
    new_kv_cache_scheme: Any | None = None,
    new_format: str | None = None,
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
    quantization_config = config.get("quantization_config")
    if not quantization_config:
        return config

    if quantization_config.get("quant_method") != "compressed-tensors":
        return config

    # Merge config_groups (update/append)
    existing_config_groups = quantization_config.get("config_groups", {})
    merged_config_groups = {**existing_config_groups}
    if new_config_groups:
        merged_config_groups.update(new_config_groups)

    # Merge ignore lists (concatenate)
    existing_ignore = quantization_config.get("ignore", []) or []
    merged_ignore = list(existing_ignore)
    if new_ignore:
        merged_ignore.extend(new_ignore)

    # Handle kv_cache_scheme (error if not identical)
    existing_kv_cache = quantization_config.get("kv_cache_scheme")
    if existing_kv_cache is not None and new_kv_cache_scheme is not None:
        if existing_kv_cache != new_kv_cache_scheme:
            raise ValueError(
                f"Cannot merge configs with different kv_cache_schemes: "
                f"{existing_kv_cache} != {new_kv_cache_scheme}"
            )
    merged_kv_cache = existing_kv_cache or new_kv_cache_scheme

    # Handle format (use mixed precision if not identical)
    existing_format = quantization_config.get("format", CompressionFormat.dense.value)
    merged_format = new_format or existing_format
    if existing_format != merged_format:
        merged_format = CompressionFormat.mixed_precision.value

    # Update quantization config with merged values
    quantization_config["config_groups"] = merged_config_groups
    quantization_config["ignore"] = merged_ignore
    quantization_config["kv_cache_scheme"] = merged_kv_cache
    quantization_config["format"] = merged_format
    # Always set quantization_status to compressed
    quantization_config["quantization_status"] = QuantizationStatus.COMPRESSED.value

    config["quantization_config"] = quantization_config
    return config
