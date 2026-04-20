# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Utilities associated with offloading functionality

| ------------------------------------------------------------------------------------------------------ | # noqa: E501
| Operation  | Without offloading support             | With offloading support                          | # noqa: E501
| ---------- | -------------------------------------- | ------------------------------------------------ | # noqa: E501
| Update     | module.name.data.copy_(new_data)       | update_offload_parameter(module, name, new_data) | # noqa: E501
| ------------------------------------------------------------------------------------------------------ | # noqa: E501
"""

from typing import Literal

import torch
from compressed_tensors.offload import (
    align_module_device,
    align_modules,
    disable_offloading,
    get_execution_device,
    get_offloaded_device,
    register_offload_module,
    remove_dispatch,
    update_offload_parameter,
)
from compressed_tensors.utils.helpers import deprecated


__all__ = [
    "get_execution_device",
    "get_offloaded_device",
    "register_offload_parameter",
    "update_offload_parameter",
    "delete_offload_parameter",
    "align_modules",
    "align_module_device",
    "register_offload_module",
    "disable_offloading",
    "remove_dispatch",
]


""" Candidates for Upstreaming """


@deprecated("module.register_parameter(name, parameter)")
def register_offload_parameter(
    module: torch.nn.Module,
    name: str,
    parameter: torch.nn.Parameter,
    offload_device: torch.device | Literal["disk"] | None = None,
):
    """
    Register a parameter to the given module which may be offloaded

    :param module: maybe offloaded module
    :param name: name of newly registered parameter
    :param parameter: parameter being registered
    :param offload_device: device on which weight will be offloaded to. If None is
        provided, then infer device from parameters on module
    """
    if offload_device == "disk":
        raise NotImplementedError("Disk offloading is not currently supported")

    module.register_parameter(name, parameter)


@deprecated("delattr(module, name)")
def delete_offload_parameter(module: torch.nn.Module, name: str):
    """
    Delete a parameter from a module which may be offloaded,
    including any metadata in _hf_hook

    :param module: maybe offloaded module
    :param name: name of parameter being deleted
    """
    delattr(module, name)
