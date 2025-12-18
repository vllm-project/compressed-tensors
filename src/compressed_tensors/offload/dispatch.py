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

from collections.abc import Container
from typing import TypeVar

import torch

from .cache import DeviceCache
from .module import OffloadedModule


ModelType = TypeVar("", bound=torch.nn.Module)


def dispatch_model(
    model: ModelType,
    device: torch.device | str,
    no_split_modules: Container[str] = tuple(),
) -> ModelType:
    if len(model._parameters) > 0:
        raise NotImplementedError(
            "Offloading is achieved by replacing modules which have direct parameters "
            "with new modules which have been wrapped. However, replacing the root "
            "can break functionality with previous implementation of `dispatch_model`. "
            "Please either remove any direct parameters to the model root, or refactor "
            "this function and its usages to use the new, wrapped root"
        )

    # each model shares a single shared cache because we have to
    # coordinate the onloading of shared tensors within the model
    cache = DeviceCache(device)

    memo = dict()
    for name, module in model.named_modules(remove_duplicate=False):
        # exclude wrapping the root
        if name == "" or isinstance(module, torch.nn.ModuleList):
            continue

        no_split = module.__class__.__name__ in no_split_modules
        offloaded_module = OffloadedModule.from_module(module, cache, no_split)

        model.set_submodule(name, offloaded_module)
        memo[module] = offloaded_module

    return model
