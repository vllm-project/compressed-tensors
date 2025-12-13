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
from compressed_tensors.quantization import QuantizationScheme, QuantizationType
from compressed_tensors.quantization.lifecycle.compressed import (
    dequantize_weight,
    quantize_weight,
)
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    register_offload_parameter,
)


__all__ = [
    "NaiveQuantizationCompressor",
    "IntQuantizationCompressor",
    "FloatQuantizationCompressor",
]


@BaseCompressor.register(name=CompressionFormat.naive_quantized.value)
class NaiveQuantizationCompressor(BaseCompressor):
    """
    Implements naive compression for quantized models. Weight of each
    quantized layer is converted from its original float type to the closest Pytorch
    type to the type specified by the layer's QuantizationArgs.
    """

    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        return (
            scheme.input_activations is None
            and scheme.weights is not None
            and scheme.output_activations is None
        )

    @classmethod
    def compress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        assert hasattr(module, "weight")
        assert hasattr(module, "weight_scale")

        with align_module_device(module):
            weight = quantize_weight(module, module.weight)
            weight = torch.nn.Parameter(weight, requires_grad=False)

            delete_offload_parameter(module, "weight")
            register_offload_parameter(module, "weight", weight)

    @classmethod
    def decompress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        with align_module_device(module):
            weight = dequantize_weight(module, module.weight)

            delete_offload_parameter(module, "weight")
            register_offload_parameter(module, "weight", weight)

    @property
    def format(self) -> CompressionFormat:
        return CompressionFormat.naive_quantized


@BaseCompressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(NaiveQuantizationCompressor):
    """
    Alias for integer quantized models
    """

    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        return (
            scheme.input_activations is None
            and scheme.weights is not None
            and scheme.weights.type == QuantizationType.INT
            and scheme.output_activations is None
        )

    @property
    def format(self) -> CompressionFormat:
        return CompressionFormat.int_quantized


@BaseCompressor.register(name=CompressionFormat.float_quantized.value)
class FloatQuantizationCompressor(NaiveQuantizationCompressor):
    """
    Alias for floating point quantized models
    """

    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        return (
            scheme.input_activations is None
            and scheme.weights is not None
            and scheme.weights.type == QuantizationType.FLOAT
            and scheme.output_activations is None
        )

    @property
    def format(self) -> CompressionFormat:
        return CompressionFormat.float_quantized
