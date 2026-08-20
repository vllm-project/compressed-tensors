# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import re

import torch
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils.moe import (
    build_expert_to_router_map,
    extract_expert_index,
    get_num_experts_config_key,
    get_num_experts_per_tok,
    load_model_config,
    renumber_expert_name,
)
from compressed_tensors.utils.safetensors_load import (
    find_config_path,
    get_checkpoint_files,
    get_weight_map,
)
from loguru import logger
from safetensors import safe_open


__all__ = ["MagnitudeExpertPruner"]


class MagnitudeExpertPruner(Converter):
    """
    Prune MoE experts based on router weight magnitude. Scores each expert ``i``
    by ``router.weight[i].abs().sum()`` (L1 magnitude) and prunes the
    lowest-scoring ``sparsity`` fraction of experts per layer, retaining the
    rest.

    Supports both 2D expert weights (one tensor per expert, e.g.
    ``model.layers.0.mlp.experts.3.gate_proj.weight``) and 3D stacked expert
    weights (``model.layers.0.mlp.experts.gate_proj.weight`` with shape
    ``[num_experts, out_features, in_features]``).

    Pruning removes expert weight tensors (or slices stacked tensors), adjusts
    router weights, and updates the model config's expert count.

    :param router_pattern: regex matching router weight tensor names
    :param expert_pattern: regex matching expert weight tensor names
    :param sparsity: fraction of experts to prune per layer, in [0, 1]
    :param num_experts_config_key: config.json attribute name for expert count
    :param retained_experts: pre-computed mapping of router_name -> retained
        expert indices (sorted). Built by :meth:`from_pretrained`.
    :param expert_to_router: mapping of expert tensor name -> associated router
        tensor name. Built by :meth:`from_pretrained`.
    :param expert_indices: for 2D experts, mapping of tensor name -> expert
        index. Empty for 3D experts.
    :param is_3d: whether expert weights are 3D stacked tensors
    """

    def __init__(
        self,
        router_pattern: str,
        expert_pattern: str,
        sparsity: float,
        num_experts_config_key: str,
        retained_experts: dict[str, list[int]],
        expert_to_router: dict[str, str],
        expert_indices: dict[str, int],
        is_3d: bool,
        num_experts_per_tok: int | None = None,
    ):
        self.router_pattern = re.compile(router_pattern)
        self.expert_pattern = re.compile(expert_pattern)
        self.sparsity = sparsity
        self.num_experts_config_key = num_experts_config_key
        self.retained_experts = retained_experts
        self.expert_to_router = expert_to_router
        self.expert_indices = expert_indices
        self.is_3d = is_3d
        self.num_experts_per_tok = num_experts_per_tok

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        router_pattern: str,
        expert_pattern: str,
        sparsity: float,
        num_experts_config_key: str | None = None,
    ) -> MagnitudeExpertPruner:
        """
        Build the converter by scanning the checkpoint for router weights,
        scoring experts, and determining which to retain.

        :param model_name_or_path: HuggingFace stub or local checkpoint path
        :param router_pattern: regex matching router weight tensor names
        :param expert_pattern: regex matching expert weight tensor names
        :param sparsity: fraction of experts to prune per layer, in [0, 1]
        :param num_experts_config_key: config.json key holding the expert count.
            If None, auto-detected from config.json.
        """
        router_re = re.compile(router_pattern)
        expert_re = re.compile(expert_pattern)

        model_files = get_checkpoint_files(model_name_or_path)
        weight_map = get_weight_map(model_files)
        config_data = load_model_config(model_files)

        # auto-detect num_experts_config_key from config.json
        if num_experts_config_key is None:
            num_experts_config_key = get_num_experts_config_key(config_data)

        # detect num_experts_per_tok for routing-floor validation
        num_experts_per_tok = get_num_experts_per_tok(config_data)

        # collect router and expert tensor names
        router_names = [n for n in weight_map if router_re.search(n)]
        expert_names = [n for n in weight_map if expert_re.search(n)]

        if not router_names:
            raise ValueError(f"No tensors matched router_pattern {router_pattern!r}")
        if not expert_names:
            raise ValueError(f"No tensors matched expert_pattern {expert_pattern!r}")

        # associate each expert tensor with a router tensor by longest common
        # dot-separated prefix
        expert_to_router = build_expert_to_router_map(expert_names, router_names)

        # detect 2D vs 3D and extract expert indices for 2D
        expert_indices, is_3d = _detect_moe_layout(
            expert_names, weight_map, model_files
        )

        if not 0 <= sparsity <= 1:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

        # load router weights and score experts
        retained_experts = _compute_retained_experts(
            router_names, weight_map, model_files, sparsity
        )

        if num_experts_per_tok is not None:
            k = min(len(v) for v in retained_experts.values())
            if k < num_experts_per_tok:
                raise ValueError(
                    f"sparsity={sparsity} retains {k} expert(s) per layer but "
                    f"num_experts_per_tok={num_experts_per_tok}; cannot prune "
                    "below the per-token routing floor"
                )

        return cls(
            router_pattern=router_pattern,
            expert_pattern=expert_pattern,
            sparsity=sparsity,
            num_experts_config_key=num_experts_config_key,
            retained_experts=retained_experts,
            expert_to_router=expert_to_router,
            expert_indices=expert_indices,
            is_3d=is_3d,
            num_experts_per_tok=num_experts_per_tok,
        )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}

        for name in list(tensors):
            tensor = tensors[name]

            if self.router_pattern.search(name):
                retained = self.retained_experts[name]
                idx = torch.tensor(retained, dtype=torch.long)
                result[name] = tensor[idx].contiguous()

            elif self.expert_pattern.search(name):
                router_name = self.expert_to_router[name]
                retained = self.retained_experts[router_name]

                if self.is_3d:
                    idx = torch.tensor(retained, dtype=torch.long)
                    result[name] = tensor.index_select(0, idx).contiguous()
                else:
                    retained_set = set(retained)
                    expert_idx = self.expert_indices[name]
                    if expert_idx not in retained_set:
                        continue
                    new_idx = retained.index(expert_idx)
                    new_name = renumber_expert_name(name, expert_idx, new_idx)
                    result[new_name] = tensor

            else:
                result[name] = tensor

        return result

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.process(tensors)

        k = min(len(v) for v in self.retained_experts.values())

        for name, tensor in out.items():
            if not self.expert_pattern.search(name):
                continue

            if self.is_3d:
                router_name = self.expert_to_router[name]
                expected = len(self.retained_experts[router_name])
                if tensor.shape[0] != expected:
                    raise ValueError(
                        f"{name}: expected first dim {expected} after prune, "
                        f"got {tensor.shape[0]}"
                    )
            else:
                expert_idx = extract_expert_index(name)
                if expert_idx is None or expert_idx >= k:
                    raise ValueError(
                        f"{name}: expert index {expert_idx} out of range for "
                        f"{k} retained experts, could be orphan"
                    )

        return out

    def update_config(
        self, config: QuantizationConfig | None, save_directory: str | None = None
    ) -> QuantizationConfig | None:
        if save_directory is not None:
            config_path = find_config_path(save_directory)
            if config_path is None:
                logger.warning(
                    f"Could not find config file in {save_directory} to update "
                    f"{self.num_experts_config_key}"
                )
            else:
                with open(config_path, "r") as f:
                    config_data = json.load(f)

                counts = {len(val) for val in self.retained_experts.values()}
                if len(counts) != 1:
                    raise ValueError(
                        f"non-uniform retained expert counts {counts}; "
                        "cannot write a single num_experts"
                    )
                num_experts = counts.pop()
                _set_nested_key(config_data, self.num_experts_config_key, num_experts)

                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2, sort_keys=True)

                logger.info(
                    f"Updated {self.num_experts_config_key} -> {num_experts} "
                    f"in {config_path}"
                )
        return config

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_nested_key(config_data: dict, key: str, value: int):
    if key in config_data:
        old = config_data[key]
        config_data[key] = value
        logger.info(f"config.json: {key}: {old} -> {value}")
    elif "text_config" in config_data and key in config_data["text_config"]:
        old = config_data["text_config"][key]
        config_data["text_config"][key] = value
        logger.info(f"config.json text_config.{key}: {old} -> {value}")
    else:
        config_data[key] = value
        logger.info(f"config.json: added {key} = {value}")


def _detect_moe_layout(
    expert_names: list[str],
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> tuple[dict[str, int], bool]:
    """
    Detect whether experts are 2D (per-expert tensors) or 3D (stacked).
    For 2D, also extract the expert index from each tensor name.

    Returns (expert_indices, is_3d).
    """
    sample_name = expert_names[0]
    sample_shard = weight_map[sample_name]
    sample_path = model_files[sample_shard]

    with safe_open(sample_path, framework="pt") as f:
        sample_shape = f.get_slice(sample_name).get_shape()

    if len(sample_shape) >= 3:
        return {}, True

    # 2D: extract expert index from each tensor name
    expert_indices: dict[str, int] = {}
    for name in expert_names:
        idx = extract_expert_index(name)
        if idx is None:
            raise ValueError(
                f"Expert tensor {name} appears to be 2D but could not extract "
                "expert index from tensor name. Expected a numeric segment "
                "following an 'expert'-containing segment (e.g. experts.3.weight)"
            )
        expert_indices[name] = idx

    return expert_indices, False


def _compute_retained_experts(
    router_names: list[str],
    weight_map: dict[str, str],
    model_files: dict[str, str],
    sparsity: float,
) -> dict[str, list[int]]:
    """
    Load each router weight, score experts by L1 magnitude
    (``weight.abs().sum(dim=-1)``), and return the retained expert indices
    (sorted) per router. Absolute values are used so positive and negative
    router weights do not cancel. The number kept is derived per router as
    ``max(1, num_experts - round(sparsity * num_experts))``.
    """
    retained: dict[str, list[int]] = {}

    # group router names by source file to minimize I/O
    routers_by_file: dict[str, list[str]] = {}
    for rname in router_names:
        shard = weight_map[rname]
        path = model_files[shard]
        routers_by_file.setdefault(path, []).append(rname)

    for path, names in routers_by_file.items():
        with safe_open(path, framework="pt") as f:
            for rname in names:
                weight = f.get_tensor(rname)
                scores = weight.abs().sum(dim=-1)
                num_experts = scores.shape[0]
                num_prune = round(sparsity * num_experts)
                k = max(1, num_experts - num_prune)
                _, top_indices = torch.topk(scores, k)
                retained[rname] = sorted(top_indices.tolist())
                logger.debug(
                    f"{rname}: retaining experts {retained[rname]} "
                    f"(dropped {num_experts - k})"
                )

    return retained
