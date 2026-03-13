# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import chain

import torch
from compressed_tensors.utils.type import TensorStateDict


__all__ = ["get_direct_state_dict", "replace_direct_state_dict"]


def get_direct_state_dict(module: torch.nn.Module) -> TensorStateDict:
    """
    Extract a state dict directly from a module's parameters and buffers.

    Returns tensor data (unwrapped from Parameter/Buffer wrappers) for all
    parameters and buffers in the module. Does not recurse into child modules.

    :param module: the module to extract state from
    :return: dict mapping parameter/buffer names to their tensor data
    """
    return {
        name: (
            tensor.data
            if isinstance(tensor, (torch.nn.Parameter, torch.nn.Buffer))
            else tensor
        )
        for name, tensor in chain(module._parameters.items(), module._buffers.items())
    }


def replace_direct_state_dict(module: torch.nn.Module, new_state_dict: TensorStateDict):
    """
    Replace a module's parameters and buffers with a new state dict.

    Removes parameters/buffers that exist in the old state but not the new state,
    and adds/updates parameters from the new state dict. All new tensors are
    added as non-trainable parameters (not buffers). Skips unchanged values
    for efficiency.

    :param module: the module to update
    :param new_state_dict: dict of new parameter/buffer values
    """
    from compressed_tensors.offload import disable_onloading, update_offload_parameter

    with disable_onloading():
        old_state_dict = get_direct_state_dict(module)

    for name in old_state_dict:
        # remove attributes that don't exist in the new state
        if name not in new_state_dict:
            delattr(module, name)

    for name, new_value in new_state_dict.items():
        # treat all new tensors as parameters (not buffers)
        new_value = torch.nn.Parameter(new_value, requires_grad=False)
        old_value = old_state_dict.get(name, None)

        if (
            old_value is not None
            and torch.is_same_size(old_value, new_value)
            and old_value.dtype == new_value.dtype
        ):
            update_offload_parameter(module, name, new_value)

        elif name in old_state_dict:
            delattr(module, name)
            module.register_parameter(name, new_value)

        else:
            module.register_parameter(name, new_value)
