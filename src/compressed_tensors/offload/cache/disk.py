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
from typing import Optional

import torch
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.utils import send_tensors


class DiskCache(CPUCache):
    """
    TODO

    Note: This cache does not currently handle propagation of in-place
    operations on the onloaded tensors. Future work could support this by
    returning a tensor subclass which references on offloaded tensor. To update
    parameters, use `compressed_tensors.offload::update_offload_parameter`
    """

    offload_device: Optional[torch.device | str] = torch.device("cpu")

    def __init__(self, onload_device: torch.device, offload_dir: Optional[str] = None):
        super().__init__(onload_device)

        # TODO: check that this gets cleaned up
        self.offload_dir = offload_dir or tempfile.gettempdir()

    def onload(self, key: torch.Tensor) -> torch.Tensor:
        return torch.load(
            self._get_save_path_from_key(key), map_location=self.onload_device
        )

    def offload(self, value: torch.Tensor) -> torch.Tensor:
        # return original tensor if onloading is disabled
        # to allow for direct parameter/buffer assignment
        if self.onloading_disabled:
            return value

        key = send_tensors(value, device="meta")
        torch.save(value, self._get_save_path_from_key(key))

        return key

    def __delitem__(self, key: torch.Tensor):
        raise NotImplementedError()

    def _get_save_path_from_key(self, key: torch.Tensor) -> str:
        return os.path.join(self.offload_dir, f"{hash(key)}.pt")
