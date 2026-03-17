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
    old_state_dict = get_direct_state_dict(module)

    for name, old_value in old_state_dict.items():
        # remove attributes that don't exist in the new state
        if name not in new_state_dict:
            delattr(module, name)

    for name, new_value in new_state_dict.items():
        # skip unchanged values
        if name not in old_state_dict or old_state_dict[name] is not new_value:
            # overwrite (not update) if param already existed
            if hasattr(module, name):
                delattr(module, name)

            # treat all new tensors as parameters (not buffers)
            setattr(module, name, torch.nn.Parameter(new_value, requires_grad=False))
