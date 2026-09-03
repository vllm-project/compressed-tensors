# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os

from compressed_tensors.utils.safetensors_load import find_config_path


__all__ = [
    "NUM_EXPERTS_CONFIG_KEYS",
    "load_model_config",
    "get_num_experts_config_key",
    "get_num_experts_per_tok",
    "common_prefix_len",
    "find_expert_index",
    "extract_expert_index",
    "renumber_expert_name",
    "build_expert_to_router_map",
]

NUM_EXPERTS_CONFIG_KEYS = [
    "num_experts",
    "num_local_experts",
    "moe_num_experts",
    "n_routed_experts",
]


def load_model_config(model_files: dict[str, str]) -> dict:
    config_path = find_config_path(os.path.dirname(next(iter(model_files.values()))))
    if config_path is None:
        raise ValueError(
            "Could not find config.json. "
            "Please specify num_experts_config_key explicitly."
        )
    with open(config_path, "r") as f:
        return json.load(f)


def get_num_experts_config_key(config_data: dict) -> str:
    for key in NUM_EXPERTS_CONFIG_KEYS:
        if key in config_data:
            return key
        if "text_config" in config_data and key in config_data["text_config"]:
            return key
    raise ValueError(
        f"Could not find any of {NUM_EXPERTS_CONFIG_KEYS} in config.json. "
        "Please specify num_experts_config_key explicitly."
    )


def get_num_experts_per_tok(config_data: dict) -> int | None:
    text = config_data.get("text_config", {})
    for key in ("top_k_experts", "num_experts_per_tok"):
        val = config_data.get(key) or text.get(key)
        if val is not None:
            return int(val)
    return None


def common_prefix_len(a: str, b: str) -> int:
    parts_a = a.split(".")
    parts_b = b.split(".")
    common = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            common += 1
        else:
            break
    return common


def find_expert_index(parts: list[str]) -> tuple[int, int] | None:
    """
    Locate the expert index in a dot-split tensor name.

    Returns ``(position, value)`` where ``position`` is the index into
    ``parts`` of the numeric segment and ``value`` is its int value, or
    ``None`` if no ``expert``-containing segment is followed by an int.
    """
    for i, part in enumerate(parts):
        if "expert" in part.lower() and i + 1 < len(parts):
            try:
                return i + 1, int(parts[i + 1])
            except ValueError:
                continue
    return None


def extract_expert_index(name: str) -> int | None:
    """
    Extract the expert index from a tensor name like
    ``model.layers.0.mlp.experts.3.gate_proj.weight``.

    Looks for a numeric dot-segment immediately following a segment that
    contains 'expert'.
    """
    found = find_expert_index(name.split("."))
    return found[1] if found else None


def renumber_expert_name(name: str, old_idx: int, new_idx: int) -> str:
    """
    Replace the expert index in a tensor name.

    ``model.layers.0.mlp.experts.5.gate_proj.weight`` with old=5, new=2
    becomes ``model.layers.0.mlp.experts.2.gate_proj.weight``.
    """
    parts = name.split(".")
    found = find_expert_index(parts)
    if found is not None and found[1] == old_idx:
        pos, _ = found
        parts[pos] = str(new_idx)
        return ".".join(parts)
    return name


def build_expert_to_router_map(
    expert_names: list[str], router_names: list[str]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for expert_name in expert_names:
        best_router = None
        best_len = -1
        for router_name in router_names:
            plen = common_prefix_len(expert_name, router_name)
            if plen > best_len:
                best_len = plen
                best_router = router_name
        if best_router is None:
            raise ValueError(
                f"Could not associate expert tensor {expert_name} with any "
                "router tensor"
            )
        mapping[expert_name] = best_router
    return mapping
