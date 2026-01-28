# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
