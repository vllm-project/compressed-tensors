# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, TypeVar


T = TypeVar("T")


__all__ = ["SearchFailureError", "max_binary_search"]


class SearchFailureError(ValueError):
    pass


def max_binary_search(
    fn: Callable[[int], T],
    cond: Callable[[T], bool],
    start: int,
    end: int,
) -> tuple[int, T]:
    best_idx = None
    best_val = None

    while start <= end:
        mid = (start + end) // 2
        val = fn(mid)

        if cond(val):
            # condition is true, search higher
            best_idx, best_val = mid, val
            start = mid + 1
        else:
            # condition is false, search lower
            end = mid - 1

    if best_idx is None:
        raise SearchFailureError()

    return best_idx, best_val
