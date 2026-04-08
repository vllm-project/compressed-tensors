# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional, TypeVar

import torch
import torch.distributed as dist
from compressed_tensors.distributed.assign import greedy_bin_packing
from compressed_tensors.distributed.utils import set_source_process
from compressed_tensors.offload.utils import as_single_threaded, module_size
from compressed_tensors.utils.module import (
    get_direct_state_dict,
    replace_direct_state_dict,
)
from tqdm import tqdm


__all__ = ["replace_module_parallel"]

T = TypeVar("T", bound=torch.nn.Module)


def replace_module_parallel(
    modules: list[T],
    apply_fn: Callable[[T], None],
    weight_fn: Callable[[T], float] = module_size,
    desc: Optional[str] = None,
):
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
    :param desc: optional description for the progress bar (shown on each rank)
    """
    from compressed_tensors.offload import disable_onloading, to_meta, OffloadCache

    _, _, assigned_rank = greedy_bin_packing(modules, dist.get_world_size(), weight_fn)

    # Count modules assigned to this rank and create progress bar
    rank = dist.get_rank()
    num_assigned = sum(int(a == rank) for a in assigned_rank.values())
    progress = tqdm(total=num_assigned, desc=desc, position=rank, disable=bool(desc))

    # Step 1 & 2: Decouple and compress on meta for non-processing ranks
    with disable_onloading():
        for module in modules:
            if assigned_rank[module] != dist.get_rank():
                to_meta(module)  # 1. remove non-processing rank pointers
                apply_fn(module)  # 2. compress on meta (prepare step 4)

    # Step 3: Apply on device for processing rank
    with as_single_threaded():
        for module in modules:
            if assigned_rank[module] == dist.get_rank():
                apply_fn(module)  # 3. compress without triggering sync
                progress.update(1)

    # Step 4: Recouple - broadcast source offload across ranks
    for module in modules:
        with disable_onloading():
            state_dict = get_direct_state_dict(module)

        # If module is not offloaded, manually broadcast tensors via object list
        if not isinstance(module._parameters, OffloadCache):
            with set_source_process(assigned_rank[module]):
                if assigned_rank[module] == dist.get_rank():
                    broadcast_obj = [state_dict]
                else:
                    broadcast_obj = [None]

                dist.broadcast_object_list(broadcast_obj, src=assigned_rank[module])
                state_dict = broadcast_obj[0]

        with set_source_process(assigned_rank[module]):
            replace_direct_state_dict(module, state_dict)  # 4. broadcast
