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
"""
Utilities associated with offloading functionality

| ------------------------------------------------------------------------------------------------------ | # noqa: E501
| Operation  | Without offloading support             | With offloading support                          | # noqa: E501
| ---------- | -------------------------------------- | ------------------------------------------------ | # noqa: E501
| Update     | module.name.data.copy_(new_data)       | update_offload_parameter(module, name, new_data) | # noqa: E501
| ------------------------------------------------------------------------------------------------------ | # noqa: E501
"""

import contextlib
from typing import Literal

import torch
from compressed_tensors.offload import (
    align_module_device,
    align_modules,
    disable_offloading,
    get_execution_device,
    get_offloaded_device,
    offload_model,
    register_offload_module,
    remove_dispatch,
    update_offload_parameter,
)
from compressed_tensors.utils.helpers import deprecated


__all__ = [
    "get_execution_device",
    "get_offloaded_device",
    "update_parameter_data",
    "register_offload_parameter",
    "update_offload_parameter",
    "delete_offload_parameter",
    "has_offloaded_params",
    "disable_hf_hook",
    "disable_offload",
    "align_modules",
    "align_module_device",
    "register_offload_module",
    "delete_offload_module",
    "offloaded_dispatch",
    "disable_offloading",
    "remove_dispatch",
    "cast_to_device",
    "offload_to_weights_map",
    "delete_from_weights_map",
]


def update_parameter_data(
    module: torch.nn.Module, new_param_data: torch.Tensor, param_name: str
):
    """
    Update the data of an existing parameter and its offload dict. Supports both
    parameters of offloaded modules and non-offloaded modules

    :param module: module containing the parameter to update
    :param new_param_data: tensor to update parameter with
    :param param_name: name of module parameter to update
    """
    update_offload_parameter(module, param_name, new_param_data)


""" Candidates for Upstreaming """


@deprecated()
def cast_to_device(device_spec: int | torch.device) -> torch.device:
    """
    Convert an integer device index or torch.device into a torch.device object.

    :param device_spec: Device index (int) or torch.device object.
                        Negative integers map to CPU.
    :return: torch.device corresponding to the given device specification.
    """
    if isinstance(device_spec, int):
        return torch.device(f"cuda:{device_spec}" if device_spec >= 0 else "cpu")
    return device_spec


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


@deprecated("compressed_tensors.offload::unwrap_offload")
@contextlib.contextmanager
def disable_hf_hook(module: torch.nn.Module):
    raise ValueError()


@deprecated("delattr(base, name)")
def delete_offload_module(base: torch.nn.Module, name: str):
    """
    Delete a submodule from a model which may contain offloading
    :param base: parent module to delete submodule from
    :param name: name of submodule on parent
    """
    delattr(base, name)


@deprecated("compressed_tensors.offload::offload_model")
def offloaded_dispatch(
    module: torch.nn.Module,
    execution_device: torch.device,
    offload_device: torch.device | Literal["disk"] | None = None,
) -> torch.nn.Module:
    """
    Dispatch a model, keeping device parameters offloaded on their current device

    :param module: module containing parameters to offload
    :param execution_device: device that modules will onload and execute on
    :param offload_device: device that module parameters will offload to
    :return: module with offloading device hooks
    """
    if offload_device is not None:
        raise ValueError(
            "Passing offload_device to offloaded_dispatch is no longer supported"
        )
    offload_model(module, execution_device)


@deprecated("compressed_tensors.offload::align_module_device")
def disable_offload(module: torch.nn.Module):
    raise ValueError()


@deprecated()
def offload_to_weights_map(*args, **kwargs):
    raise ValueError()


@deprecated()
def delete_from_weights_map(*args, **kwargs):
    raise ValueError()


@deprecated()
def has_offloaded_params(module: torch.nn.Module) -> bool:
    """
    Checks if a module has offloaded parameters by checking if the given module has a
    AlignDevicesHook attached with offloading enabled

    Args:
        module (`torch.nn.Module`): The module to check for an offload hook.

    Returns:
        bool: `True` if the module has an offload hook and offloading is enabled,
        `False` otherwise.
    """
    return False
