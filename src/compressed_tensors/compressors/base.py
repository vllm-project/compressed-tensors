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

from abc import ABC, abstractmethod

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.registry import RegistryMixin
from torch.nn import Module


__all__ = ["BaseCompressor", "ALL_FORMAT_NAMES"]


# order determines the order of auto-inference
ALL_FORMAT_NAMES = [
    "nvfp4-pack-quantized",
    "mxfp4-pack-quantized",
    "marlin-24",
    "pack-quantized",
    "float-quantized",
    "int-quantized",
    "naive-quantized",
]


class BaseCompressor(RegistryMixin, ABC):
    @staticmethod
    @abstractmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def compress_module(cls, module: Module) -> Module:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def decompress_module(cls, module: Module) -> Module:
        raise NotImplementedError()

    @property
    @abstractmethod
    def format(self) -> CompressionFormat:
        raise NotImplementedError()
