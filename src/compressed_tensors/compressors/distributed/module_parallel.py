# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, TypeVar

import torch
import torch.distributed as dist
from compressed_tensors.compressors.distributed.assign import greedy_bin_packing
from compressed_tensors.offload import disable_onloading, to_meta
from compressed_tensors.offload.dist_utils import as_single_threaded, set_main_process
from compressed_tensors.utils.module import (
    get_direct_state_dict,
    replace_direct_state_dict,
)


__all__ = ["apply_module_parallel"]

T = TypeVar("T", bound=torch.nn.Module)


def apply_module_parallel(
    modules: list[T],
    apply_fn: Callable[[T], None],
    weight_fn: Callable[[T], float],
) -> None:
    """Apply a function to modules in parallel across distributed ranks.

    Distributes modules across ranks using greedy bin packing, then applies
    the function to each module on its assigned rank. Non-processing ranks
    temporarily move their modules to meta device to avoid increasing peak
    memory usage during compression.

    This implements the 4-step algorithm:
    1. Decouple: Move non-processing rank modules to meta device
    2. Compress On Meta: Apply function on meta device (prepare for step 4)
    3. Compress On Device: Processing rank applies function without sync
    4. Recouple: Broadcast offload pointer information across ranks

    :param modules: list of modules to process
    :param apply_fn: function to apply to each module
    :param weight_fn: function that returns the weight/size of a module
        for load balancing across ranks
    """
    _, _, assigned_rank = greedy_bin_packing(modules, dist.get_world_size(), weight_fn)

    # Step 1 & 2: Decouple and compress on meta for non-processing ranks
    with disable_onloading():
        for module in modules:
            if assigned_rank[module] != dist.get_rank():
                to_meta(module)  # 1. remove non-processing rank pointers
                apply_fn(module)  # 2. compress on meta (prepare step 4)

    # Step 3: Compress on device for processing rank
    with as_single_threaded():
        for module in modules:
            if assigned_rank[module] == dist.get_rank():
                apply_fn(module)  # 3. compress without triggering sync

    # Step 4: Recouple - broadcast source offload across ranks
    for module in modules:
        with disable_onloading():
            state_dict = get_direct_state_dict(module)
        with set_main_process(assigned_rank[module]):
            replace_direct_state_dict(module, state_dict)  # 4. broadcast
