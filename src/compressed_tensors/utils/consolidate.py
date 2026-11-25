# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from itertools import product


__all__ = ["consolidate_module_names"]


def _get_fqn_spec(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Spec of a fully qualified name.

    A numeric component, such as a layer index, becomes ``#``.
    """
    return tuple("#" if part.isdigit() else part for part in parts)


def _fqn_sort_key(parts: tuple[str, ...]) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _is_index_rectangle(fqns: set[tuple[str, ...]], columns: list[int]) -> bool:
    """True when every combination of the numeric columns is present.

    ``{(0, 0), (0, 1), (1, 0), (1, 1)}`` is a rectangle. ``{(0, 0), (1, 1)}``
    is a diagonal: compressing that per column would also match the other
    two corners.
    """
    if not columns:
        return False
    pools = [sorted({fqn[column] for fqn in fqns}, key=int) for column in columns]
    present = {tuple(fqn[column] for column in columns) for fqn in fqns}
    return present == set(product(*pools))


def _fqn_alternation(fqns: set[tuple[str, ...]]) -> str:
    ordered = sorted(fqns, key=_fqn_sort_key)
    inner = "|".join(re.escape(".".join(parts)) for parts in ordered)
    return f"re:^(?:{inner})\\Z"


def _rectangle_pattern(fqns: set[tuple[str, ...]]) -> str:
    """One anchored regex. Each numeric slot lists only the indexes present."""
    spec = _get_fqn_spec(next(iter(fqns)))
    pieces: list[str] = []
    for index, token in enumerate(spec):
        if token != "#":
            pieces.append(re.escape(token))
            continue
        values = sorted({fqn[index] for fqn in fqns}, key=int)
        if len(values) == 1:
            pieces.append(re.escape(values[0]))
            continue
        choice = "|".join(re.escape(value) for value in values)
        pieces.append(f"(?:{choice})")
    return "re:^" + r"\.".join(pieces) + r"\Z"


def _pattern_for_name_group(fqns: set[tuple[str, ...]]) -> str:
    if len(fqns) == 1:
        return ".".join(next(iter(fqns)))
    spec = _get_fqn_spec(next(iter(fqns)))
    columns = [index for index, token in enumerate(spec) if token == "#"]
    if _is_index_rectangle(fqns, columns):
        return _rectangle_pattern(fqns)
    return _fqn_alternation(fqns)


def consolidate_module_names(names: list[str]) -> list[str]:
    """Fold fully qualified names that share a spec into one regex.

    Names are grouped by spec. A numeric component is a wildcard in the
    spec, so these two are one group::

        model.layers.0.mlp.gate
        model.layers.1.mlp.gate

    What a group becomes:

    * One name stays a literal: ``model.layers.0.mlp.down_proj``.
    * A rectangle of indexes becomes one anchored regex. Each numeric
      position lists only the indexes in the group::

        model.layers.0.mlp.gate
        model.layers.1.mlp.gate
        -> re:^model\\.layers\\.(?:0|1)\\.mlp\\.gate\\Z

      A full rectangle of layers and experts becomes::

        layers.{0,1}.experts.{0,1}.gate
        -> re:^layers\\.(?:0|1)\\.experts\\.(?:0|1)\\.gate\\Z

      ``\\d+`` is not used. A missing index may belong to a module that
      should not match this pattern.
    * A set that is not a rectangle stays an alternation of the full names,
      so a diagonal does not also match the other corners::

        layers.0.experts.0.gate
        layers.1.experts.1.gate
        -> re:^(?:layers\\.0\\.experts\\.0\\.gate|layers\\.1\\.experts\\.1\\.gate)\\Z

    An entry that is already ``re:`` is copied through. A group is emitted
    where its first concrete name appeared, so the rest of the list stays
    in order.

    Matchers use ``re.match``, which does not require the pattern to consume
    the whole name, so every generated pattern is anchored with ``^`` and
    ``\\Z``.
    """
    groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    # A regex stays where it is. A spec is emitted where its first concrete
    # name appeared.
    order: list[str | tuple[str, ...]] = []
    for name in names:
        if name.startswith("re:"):
            order.append(name)
            continue
        if not name:
            continue
        parts = tuple(name.split("."))
        spec = _get_fqn_spec(parts)
        if spec not in groups:
            groups[spec] = []
            order.append(spec)
        groups[spec].append(parts)

    consolidated: list[str] = []
    for entry in order:
        if isinstance(entry, str):
            consolidated.append(entry)
            continue
        consolidated.append(_pattern_for_name_group(set(groups[entry])))
    return consolidated
