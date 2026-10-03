# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

from compressed_tensors.utils.moe import find_expert_index
from safetensors import safe_open


__all__ = [
    "OutputTensor",
    "build_output_tensors",
    "replace_segment",
    "repack_ignore",
    "validate_linearized",
]

GATE_PROJ = "gate_proj"
UP_PROJ = "up_proj"
GATE_UP_PROJ = "gate_up_proj"


@dataclass
class OutputTensor:
    """
    A single stacked output tensor and the per-expert source tensors it is built
    from. ``sources`` holds one ordered (by expert index) member list per source
    projection: one list for a plain stack, two (gate, up) for a fused output.
    """

    name: str
    sources: list[list[str]]

    @property
    def anchor(self) -> str:
        # the lowest-index tensor of the first projection; the output's primary
        # weight, whose shard the packed result is written to
        return self.sources[0][0]

    @property
    def members(self) -> list[str]:
        return [name for group in self.sources for name in group]


def replace_segment(name: str, old: str, new: str) -> str | None:
    """Replace the first dot-segment equal to ``old`` with ``new``, or None."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == old:
            parts[i] = new
            return ".".join(parts)
    return None


def build_output_tensors(
    groups: dict[str, list[str]], fuse_gate_up: bool
) -> list[OutputTensor]:
    """
    Turn the per-projection groups into output tensors. With ``fuse_gate_up``,
    each ``gate_proj`` group is paired with its ``up_proj`` sibling into a single
    ``gate_up_proj`` output; all other groups stack on their own.
    """
    up_of_gate: dict[str, str] = {}
    if fuse_gate_up:
        for name in groups:
            up_name = replace_segment(name, GATE_PROJ, UP_PROJ)
            if up_name in groups:
                up_of_gate[name] = up_name
    fused_up = set(up_of_gate.values())

    output_tensors: list[OutputTensor] = []
    for name, members in groups.items():
        if name in fused_up:
            continue  # emitted as part of its gate partner
        if name in up_of_gate:
            up_members = groups[up_of_gate[name]]
            if len(members) != len(up_members):
                raise ValueError(
                    f"Cannot fuse {name!r} ({len(members)} experts) with "
                    f"{up_of_gate[name]!r} ({len(up_members)} experts)"
                )
            fused_name = replace_segment(name, GATE_PROJ, GATE_UP_PROJ)
            output_tensors.append(OutputTensor(fused_name, [members, up_members]))
        else:
            output_tensors.append(OutputTensor(name, [members]))
    return output_tensors


def repack_ignore(ignore: list[str]) -> list[str]:
    """
    Rewrite each per-expert-module ignore entry to its packed name, dropping the
    expert-index segment and everything after it, and deduplicating the result
    (an expert's gate/up/down projections all collapse to the same module).
    Regex (``re:``) and non-expert entries pass through unchanged.
    """
    result: list[str] = []
    for entry in ignore:
        if not entry.startswith("re:"):
            found = find_expert_index(entry.split("."))
            if found is not None:
                entry = ".".join(entry.split(".")[: found[0]])
        if entry not in result:
            result.append(entry)
    return result


def validate_linearized(groups, weight_map, model_files):
    """Guard against experts that are already stacked (3D)."""
    anchor = next(iter(groups.values()))[0]
    with safe_open(model_files[weight_map[anchor]], framework="pt") as f:
        ndim = len(f.get_slice(anchor).get_shape())
    if ndim >= 3 and anchor.endswith("weight_packed"):
        raise ValueError(
            f"Expert tensor {anchor} is already {ndim}D; experts appear to be "
            "packed already, nothing to repack."
        )
