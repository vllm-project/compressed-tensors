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

from dataclasses import fields, is_dataclass
from typing import TypeVar

import torch
from loguru import logger


__all__ = ["send_tensors", "get_module_device"]

T = TypeVar("T")


def send_tensors(value: T, *args, **kwargs) -> T:
    match value:
        case torch.nn.Parameter():
            data = value.to(*args, **kwargs)
            return torch.nn.Parameter(data, requires_grad=value.requires_grad)
        case torch.Tensor():
            return value.to(*args, **kwargs)
        case list():
            return [send_tensors(v, *args, **kwargs) for v in value]
        case tuple():
            return tuple(send_tensors(v, *args, **kwargs) for v in value)
        case dict():
            return {k: send_tensors(v, *args, **kwargs) for k, v in value.items()}
        case _ if is_dataclass(value):
            for field in fields(value):
                v = getattr(value, field.name)
                setattr(value, field.name, send_tensors(v, *args, **kwargs))
            return value
        case _:
            return value


def get_module_device(module: torch.nn.Module) -> torch.device:
    tensor = next(module.parameters(), next(module.buffers(), None))
    if tensor is not None:
        return tensor.device
    else:
        logger.warning(
            f"Unable to get execution device of {module}, falling back to CPU"
        )
        return torch.device("cpu")
