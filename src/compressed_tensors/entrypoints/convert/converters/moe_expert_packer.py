# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re

import torch
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils.moe import find_expert_index
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)
from loguru import logger

from .moe_expert_packer_helpers import (
    OutputTensor,
    build_output_tensors,
    repack_ignore,
    repack_target,
    validate_linearized,
)


__all__ = ["MoEExpertPacker"]


class MoEExpertPacker(Converter):
    """
    Repack per-expert (2D "linearized") MoE weights into stacked 3D tensors.

    Some checkpoints store each expert as its own module, so an ``E``-expert MoE
    layer produces ``E`` separate 2D tensors per projection::

        model.layers.0.mlp.experts.{0..E-1}.gate_proj.weight_packed  [out, in]

    Fused/grouped MoE kernels instead expect a single 3D tensor per projection,
    with the expert index promoted to a leading dimension::

        model.layers.0.mlp.experts.gate_proj.weight_packed           [E, out, in]

    Every per-expert tensor is stacked, including the accompanying quantization
    params (``weight_scale``, ``weight_global_scale``, ...), so a ``[1]`` scale
    becomes ``[E, 1]``. The quantization format is unchanged. The config
    ``ignore`` list is rewritten: entries naming per-expert modules are collapsed
    to the packed expert module (see :meth:`update_config`).

    With ``fuse_gate_up`` (the default), each expert's ``gate_proj`` and
    ``up_proj`` are fused into a single ``gate_up_proj`` output, concatenated
    along the output-feature dim: ``[E, 2*out, in]``.

    :param expert_pattern: regex matching per-expert tensor names
    :param groups: stacked (index-stripped) name -> per-expert source names,
        ordered by expert index. Built by :meth:`from_pretrained`.
    :param fuse_gate_up: whether to fuse ``gate_proj`` and ``up_proj``
    """

    def __init__(
        self,
        expert_pattern: str,
        groups: dict[str, list[str]],
        fuse_gate_up: bool = True,
    ):
        self.expert_pattern = re.compile(expert_pattern)
        self.groups = groups
        self.fuse_gate_up = fuse_gate_up

        self.output_tensors = build_output_tensors(groups, fuse_gate_up)
        self._by_anchor = {out.anchor: out for out in self.output_tensors}
        self._members = {name for out in self.output_tensors for name in out.members}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        expert_pattern: str = r"\.experts\.\d+\.",
        fuse_gate_up: bool = True,
    ) -> MoEExpertPacker:
        """
        Build the converter by scanning the checkpoint weight map for linearized
        experts and grouping the per-expert tensors that stack together.

        Tensors matching ``expert_pattern`` are grouped by their stacked name --
        the tensor name with the numeric expert-index segment removed -- and
        ordered within each group by ascending expert index.

        :param model_name_or_path: HuggingFace stub or local checkpoint path
        :param expert_pattern: regex matching per-expert tensor names
        :param fuse_gate_up: whether to fuse ``gate_proj`` and ``up_proj``
        """
        expert_re = re.compile(expert_pattern)
        model_files = get_checkpoint_files(model_name_or_path)
        weight_map = get_weight_map(model_files)

        expert_names = [n for n in weight_map if expert_re.search(n)]
        if not expert_names:
            raise ValueError(f"No tensors matched expert_pattern {expert_pattern!r}")

        # group per-expert names by stacked name, tracking each expert index so
        # members can be ordered before stacking
        grouped: dict[str, list[tuple[int, str]]] = {}
        for name in expert_names:
            found = find_expert_index(name.split("."))
            if found is None:
                raise ValueError(
                    f"Expert tensor {name} matched expert_pattern but no numeric "
                    "expert-index segment could be extracted (e.g. experts.3.)"
                )
            pos, idx = found
            parts = name.split(".")
            del parts[pos]
            grouped.setdefault(".".join(parts), []).append((idx, name))

        groups: dict[str, list[str]] = {}
        for stacked_name, members in grouped.items():
            members.sort()
            indices = [idx for idx, _ in members]
            if indices != list(range(len(indices))):
                raise ValueError(
                    f"Expert group {stacked_name!r} has non-contiguous indices "
                    f"{indices}; a source tensor may be missing or duplicated."
                )
            groups[stacked_name] = [name for _, name in members]

        validate_linearized(groups, weight_map, model_files)

        logger.info(
            f"Found {len(groups)} expert group(s) to repack across "
            f"{len(next(iter(groups.values())))} experts"
            + (" (fusing gate_proj + up_proj)" if fuse_gate_up else "")
        )
        return cls(expert_pattern, groups, fuse_gate_up)

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()

        # Per-expert tensors of one projection are stacked, and a stacked weight
        # also needs a single dense shape. Check both up front so a heterogeneous
        # checkpoint fails with the offending tensor named, rather than with
        # torch's "stack expects each tensor to be equal size".
        self._validate_expert_shapes(tensors)

        # build each output tensor whose anchor is present; the pipeline loads all
        # of an anchor's source tensors alongside it (its declared dependencies)
        for output_tensor in self.output_tensors:
            if output_tensor.anchor not in tensors:
                continue
            missing = [m for m in output_tensor.members if m not in tensors]
            if missing:
                raise ValueError(
                    f"Cannot pack {output_tensor.name!r}: "
                    f"missing expert tensors {missing}"
                )
            if output_tensor.name.endswith(".weight_shape"):
                result[output_tensor.name] = _packed_weight_shape(
                    output_tensor, tensors
                )
                consumed.update(output_tensor.members)
                continue
            # stack each projection over experts, then concatenate the
            # projections along the output-feature dim (a no-op when there is one)
            stacked = [
                torch.stack([tensors[m] for m in group])
                for group in output_tensor.sources
            ]
            result[output_tensor.name] = torch.cat(stacked, dim=1).contiguous()
            consumed.update(output_tensor.members)

        for name, tensor in tensors.items():
            if name in consumed:
                continue
            if name in self._members:
                raise ValueError(
                    f"Expert tensor {name} was loaded without its output anchor; "
                    "cannot repack it in isolation."
                )
            result[name] = tensor

        return result

    def _validate_expert_shapes(self, tensors: dict[str, torch.Tensor]) -> None:
        """
        Ensure the experts of each output tensor agree on their shape.

        :param tensors: per-expert source tensors, keyed by their original names
        :raises ValueError: if any group mixes experts of different shapes
        """
        for output_tensor in self.output_tensors:
            if output_tensor.anchor not in tensors:
                continue
            for group in output_tensor.sources:
                present = [name for name in group if name in tensors]
                if len(present) < 2:
                    continue
                shape = tensors[present[0]]
                for name in present[1:]:
                    # shapes must match; values are per-expert and legitimately
                    # differ (each expert has its own scales and weights)
                    if tuple(tensors[name].shape) != tuple(shape.shape):
                        raise ValueError(
                            f"Cannot pack {output_tensor.name!r}: {name} has "
                            f"shape {tensors[name].tolist()}, but "
                            f"{present[0]} has {shape.tolist()}"
                        )

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        # weights are only regrouped, so the scheme is preserved; but ignore
        # entries naming now-removed per-expert modules are collapsed to the
        # packed expert module (e.g. `...experts.0.down_proj` -> `...experts`),
        # and exact per-expert targets are remapped to their packed names
        # (e.g. `...experts.0.gate_proj` -> `...experts.gate_up_proj`)
        if config is not None:
            if config.ignore:
                config.ignore = repack_ignore(config.ignore)
            for scheme in config.config_groups.values():
                targets = getattr(scheme, "targets", None)
                if isinstance(targets, list):
                    scheme.targets = list(
                        dict.fromkeys(
                            repack_target(t, self.fuse_gate_up) for t in targets
                        )
                    )
        return config

    def get_dependencies(self, weight_name: str) -> set[str]:
        output_tensor = self._by_anchor.get(weight_name)
        if output_tensor is None:
            # non-anchors (and unrelated tensors) declare nothing, so they are
            # never treated as primary weights
            return set()
        return set(output_tensor.members) - {weight_name}


def _packed_weight_shape(
    output_tensor: OutputTensor, tensors: dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    ``weight_shape`` records the dense shape of its weight, so rebuild it as the
    shape of the stacked weight, ``[E, sum(out), in]``, which is what compressing
    a 3D weight stores. Stacking it like the other tensors would give ``[E, 2]``
    (``[E, 4]`` when gate/up are fused) instead, and
    :meth:`PackedQuantizationCompressor.decompress` reads the former.

    :param output_tensor: the ``weight_shape`` output being packed
    :param tensors: per-expert source tensors, keyed by their original names
    :raises ValueError: if experts of one projection disagree on their shape, or
        fused projections disagree on the input size
    """
    anchor = tensors[output_tensor.anchor]
    if anchor.is_meta:
        # validation pass: values are unknown, only the output layout matters
        return torch.empty(3, dtype=anchor.dtype, device="meta")

    out_features = 0
    in_features = None
    for group in output_tensor.sources:
        shape = tensors[group[0]]
        if in_features is not None and int(shape[-1]) != in_features:
            raise ValueError(
                f"Cannot pack {output_tensor.name!r}: fused projections have "
                "different input sizes"
            )
        out_features += int(shape[0])
        in_features = int(shape[-1])

    num_experts = len(output_tensor.sources[0])
    return torch.tensor([num_experts, out_features, in_features], dtype=anchor.dtype)
