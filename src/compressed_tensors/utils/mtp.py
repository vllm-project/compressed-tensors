# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

from compressed_tensors.base import QUANTIZATION_CONFIG_NAME
from compressed_tensors.utils.safetensors_load import (
    _fetch_and_save_prefix_tensors,
    find_config_path,
    get_weight_mappings,
    update_safetensors_index,
)


__all__ = ["save_mtp_tensors_to_checkpoint"]


def save_mtp_tensors_to_checkpoint(
    source_model: str,
    dest_dir: str,
    mtp_prefix: str = "mtp",
    shard_name: str = "model_mtp.safetensors",
):
    """
    Extracts MTP (Multi-Token Prediction) tensors from a source model checkpoint
    and saves them into a destination checkpoint directory. Updates the
    safetensors index to include the new MTP shard and updates the
    quantization_config ignore list in config.json so inference engines skip
    quantization for MTP layers.

    MTP layers are not quantized and are excluded from the model's state dict
    during quantization (e.g. via _keys_to_ignore_on_load_unexpected). This
    function copies them as-is from the original checkpoint so they are present
    in the final saved checkpoint.

    :param source_model: local path or HuggingFace stub of the original
        (unquantized) model to extract MTP tensors from
    :param dest_dir: path to the quantized checkpoint directory to save MTP
        tensors into
    :param mtp_prefix: key prefix used to identify MTP tensors, defaults to
        "mtp"
    :param shard_name: filename for the new shard written into dest_dir,
        defaults to "model_mtp.safetensors"
    """
    # Extract MTP tensors from the original checkpoint and save them as a new
    # shard in dest_dir. MTP layers are not part of the quantized model so they
    # must be carried over as-is.
    mtp_tensors = _fetch_and_save_prefix_tensors(
        source_model, mtp_prefix, dest_dir, shard_name
    )

    # Build weight_map from existing index or single-shard file, then add MTP entries.
    # update_safetensors_index will create the index file if it doesn't exist yet.
    weight_map = {
        k: os.path.basename(v) for k, v in get_weight_mappings(dest_dir).items()
    }

    weight_map.update({key: shard_name for key in mtp_tensors})
    total_size = sum(
        os.path.getsize(os.path.join(dest_dir, s)) for s in set(weight_map.values())
    )
    update_safetensors_index(dest_dir, total_size, weight_map)

    # Update quantization_config.ignore in config.json so inference engines
    # know not to apply quantization to MTP layers
    config_path = find_config_path(dest_dir)
    if config_path is not None:
        with open(config_path, "r") as f:
            config = json.load(f)

        quant_config = config.get(QUANTIZATION_CONFIG_NAME)
        if quant_config is not None:
            ignore_list = quant_config.get("ignore") or []
            mtp_ignore_pattern = f"re:^{mtp_prefix}.*"
            if mtp_ignore_pattern not in ignore_list:
                ignore_list.append(mtp_ignore_pattern)
                quant_config["ignore"] = ignore_list
                config[QUANTIZATION_CONFIG_NAME] = quant_config
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
