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

import os
import tempfile
from typing import TYPE_CHECKING, Optional

import torch
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.utils import send_tensors, to_tensor
from safetensors import safe_open
from safetensors.torch import save_file


if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType


class DiskCache(OffloadCache):
    """
    Handles offloading and onloading tensors from/to disk.

    Tensors usually start as a key in safetensors file, converted by (TODO NAME).
    New or updated tensors are written to new safetensors files in `offload_dir`.

    Tensors are stored in memory as meta tensors. The mapping between offloaded meta
    tensors and their locations on disk is defined by `index`.
    """

    offload_device: Optional[torch.device | str] = "disk"

    # offloaded tensors -> weight info
    index: dict[torch.Tensor, dict[str, str]] = dict()

    # directory where new tensors are written to
    offload_dir: str
    _new_file_prefix = "ct_disk_cache"

    def __init__(self, onload_device: torch.device, offload_dir: Optional[str] = None):
        super().__init__(onload_device)
        self.offload_dir = offload_dir or tempfile.mkdtemp()

    def onload(self, offloaded: torch.Tensor | None) -> torch.Tensor:
        """
        Onload a tensor from disk/meta to device

        :param offloaded: meta tensor to onload
        :return: device tensor, read from disk
        """
        weight_info = self.index[offloaded]
        device = _get_safe_open_device(self.onload_device)

        with safe_open(
            weight_info["safetensors_file"], framework="pt", device=device
        ) as file:
            onloaded = file.get_tensor(weight_info["weight_name"])
            onloaded = to_tensor(onloaded, offloaded)
            onloaded = onloaded.to(getattr(torch, weight_info["dtype"]))
            return onloaded

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """
        Offload a tensor to disk by writing a new safetensors file

        :param tensor: tensor on any device
        :return: meta tensor representing the offloaded tensor
        """
        if tensor is None:
            return None

        if tensor.device.type == "meta":
            assert tensor in self.index
            return tensor

        offloaded = send_tensors(tensor, device="meta")

        file_name = f"{self._new_file_prefix}{id(tensor)}.safetensors"
        file_path = os.path.join(self.offload_dir, file_name)
        self.index[offloaded] = {
            "safetensors_file": file_path,
            "weight_name": "weight",
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }

        save_file({"weight": tensor}, file_path)
        return offloaded

    def __delitem__(self, key: str):
        """
        Remove the offloaded tensor associated with `key`. Any references to its
        onloaded tensors held by this class are invalidated.

        :param key: name of tensor to invalidate
        """
        offloaded = self.offloaded_values[key]
        file_path = self.index[offloaded]["safetensors_file"]
        if os.path.basename(file_path).startswith(self._new_file_prefix):
            os.remove(file_path)
        del self.index[offloaded]
        super().__delitem__(key)


def _get_safe_open_device(device: "DeviceLikeType") -> str | int:
    """
    `safetensors.safe_open` does not accept `torch.device` as argument, so
    we must convert from torch.device to a string, while considering "cuda" resolution

    :param device: torch device to convert
    :return: device argument to `safetensors.safe_open`
    """
    device = torch.device(device)
    if device.type in ("cuda"):
        if device.index is None:
            return torch.cuda.current_device()
        else:
            return device.index
    else:
        return device.type
