from collections import defaultdict
from typing import Mapping, TypeVar

__all__ = ["invert_mapping", "get_unmatched_names"]

KeyType = TypeVar("K")
ValueType = TypeVar("V")


def invert_mapping(
    mapping: Mapping[KeyType, ValueType],
) -> dict[ValueType, list[KeyType]]:
    inverse = defaultdict(list)

    for key, value in mapping.items():
        inverse[value].append(key)

    return inverse
