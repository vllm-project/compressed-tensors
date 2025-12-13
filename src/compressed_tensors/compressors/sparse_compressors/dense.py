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

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme


@BaseCompressor.register(name=CompressionFormat.dense.value)
class DenseCompressor(BaseCompressor):
    """
    Identity compressor for dense models, returns the original state_dict
    """

    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        return True

    @classmethod
    def compress_module(cls, module: torch.nn.Module) -> torch.nn.Module:
        return module

    @classmethod
    def decompress_module(cls, module: torch.nn.Module) -> torch.nn.Module:
        return module

    @property
    def format(self) -> CompressionFormat:
        return CompressionFormat.dense
