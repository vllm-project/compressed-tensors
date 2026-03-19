# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from typing import Mapping, TypeVar


__all__ = ["invert_mapping"]

KeyType = TypeVar("K")
ValueType = TypeVar("V")


def invert_mapping(
    mapping: Mapping[KeyType, ValueType],
) -> dict[ValueType, list[KeyType]]:
    inverse = defaultdict(list)

    for key, value in mapping.items():
        inverse[value].append(key)

    return inverse
