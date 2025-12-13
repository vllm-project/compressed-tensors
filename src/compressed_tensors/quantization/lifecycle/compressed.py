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
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.quant_config import QuantizationStatus
from torch.nn import Linear


__all__ = [
    "compress_quantized_weights",
    "quantize_weight",
    "dequantize_weight",
]


def compress_quantized_weights(module: Linear):
    """
    Quantizes the module weight representation to use fewer bits in memory

    apply to full model with `model.apply(compress_quantized_weights)`

    :param module: module to compress to quantized representation
    """
    scheme = getattr(module, "quantization_scheme", None)
    status = getattr(module, "quantization_status", None)
    if not scheme or not scheme.weights or status >= QuantizationStatus.COMPRESSED:
        return

    module.weight.requires_grad = False
    module.weight.data = quantize_weight(module, module.weight)
    module.quantization_status = QuantizationStatus.COMPRESSED


def quantize_weight(module: Linear, weight: torch.Tensor) -> torch.Tensor:
    return quantize(
        weight,
        weight_scale=module.weight_scale,
        zero_point=module.weight_zero_point,
        args=module.quantization_scheme.args,
        g_idx=getattr(module, "weight_g_idx", None),
        global_scale=getattr(module, "global_scale", None),
    )


def dequantize_weight(module: Linear, weight: torch.Tensor) -> torch.Tensor:
    return dequantize(
        weight,
        weight_scale=module.weight_scale,
        zero_point=module.weight_zero_point,
        args=module.quantization_scheme.args,
        g_idx=getattr(module, "weight_g_idx", None),
        global_scale=getattr(module, "global_scale", None),
    )
