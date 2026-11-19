# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from itertools import product


__all__ = ["consolidate_module_names"]


def _path_shape(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Path shape. A numeric component, such as a layer index, becomes '#'."""
    return tuple("#" if part.isdigit() else part for part in parts)


def _path_sort_key(parts: tuple[str, ...]) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _is_index_rectangle(paths: set[tuple[str, ...]], columns: list[int]) -> bool:
    """True when every combination of the numeric columns is present.

    ``{(0, 0), (0, 1), (1, 0), (1, 1)}`` is a rectangle. ``{(0, 0), (1, 1)}``
    is a diagonal: compressing that per column would also match the other
    two corners.
    """
    if not columns:
        return False
    pools = [sorted({path[column] for path in paths}, key=int) for column in columns]
    present = {tuple(path[column] for column in columns) for path in paths}
    return present == set(product(*pools))


def _full_path_alternation(paths: set[tuple[str, ...]]) -> str:
    ordered = sorted(paths, key=_path_sort_key)
    inner = "|".join(re.escape(".".join(parts)) for parts in ordered)
    return f"re:^(?:{inner})\\Z"


def _rectangle_pattern(paths: set[tuple[str, ...]]) -> str:
    """One anchored regex. Each numeric slot lists only the indexes present."""
    shape = _path_shape(next(iter(paths)))
    pieces: list[str] = []
    for index, token in enumerate(shape):
        if token != "#":
            pieces.append(re.escape(token))
            continue
        values = sorted({path[index] for path in paths}, key=int)
        if len(values) == 1:
            pieces.append(re.escape(values[0]))
            continue
        choice = "|".join(re.escape(value) for value in values)
        pieces.append(f"(?:{choice})")
    return "re:^" + r"\.".join(pieces) + r"\Z"


def _pattern_for_name_group(paths: set[tuple[str, ...]]) -> str:
    if len(paths) == 1:
        return ".".join(next(iter(paths)))
    shape = _path_shape(next(iter(paths)))
    columns = [index for index, token in enumerate(shape) if token == "#"]
    if _is_index_rectangle(paths, columns):
        return _rectangle_pattern(paths)
    return _full_path_alternation(paths)


def consolidate_module_names(names: list[str]) -> list[str]:
    """Fold module names that share a path shape into one regex.

    Names are grouped by path shape. A numeric component is a wildcard in
    the shape, so these two are one group::

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
    * A set that is not a rectangle stays an alternation of the full paths,
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
    # A regex stays where it is. A shape is emitted where its first concrete
    # name appeared.
    order: list[str | tuple[str, ...]] = []
    for name in names:
        if name.startswith("re:"):
            order.append(name)
            continue
        if not name:
            continue
        parts = tuple(name.split("."))
        shape = _path_shape(parts)
        if shape not in groups:
            groups[shape] = []
            order.append(shape)
        groups[shape].append(parts)

    consolidated: list[str] = []
    for entry in order:
        if isinstance(entry, str):
            consolidated.append(entry)
            continue
        consolidated.append(_pattern_for_name_group(set(groups[entry])))
    return consolidated
