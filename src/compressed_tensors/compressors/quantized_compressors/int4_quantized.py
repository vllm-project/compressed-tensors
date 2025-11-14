# Copyright (c) 2025 - present / Neuralmagic, Inc. All Rights Reserved.
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
import math
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.quantized_compressors.base import (
    BaseQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import can_quantize
from torch import Tensor


__all__ = ["Int4PackedQuantizationCompressor"]

def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "the last dim size of int4_values_interleaved tensor must be even."
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)

    return packed_tensor.to(torch.int8)


def unpack_int4_values_to_int8(packed_tensor: torch.Tensor) -> torch.Tensor:
    low_nibbles  =  packed_tensor & 0x0F
    high_nibbles = (packed_tensor >> 4) & 0x0F

    out_shape = list(packed_tensor.shape)
    out_shape[-1] *= 2
    unpacked = torch.empty(out_shape, dtype=torch.int8, device=packed_tensor.device)

    unpacked[..., 0::2] = low_nibbles
    unpacked[..., 1::2] = high_nibbles
    return unpacked

def pack_interleave(ref_weight):
    n, k = ref_weight.shape[0], ref_weight.shape[1]
    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((n, k // 2)).view(torch.int8)
    w_q = w_q.contiguous()
    return w_q

def unpack_interleave(w_q: torch.Tensor) -> torch.Tensor:
    n, k_half = w_q.shape
    packed = w_q.contiguous().view(n, k_half)
    ref_weight_int8 = unpack_int4_values_to_int8(packed)
    return ref_weight_int8


@BaseCompressor.register(name=CompressionFormat.int4_quantized.value)
class Int4PackedQuantizationCompressor(BaseQuantizationCompressor):
    """
    Compresses a quantized model by packing every eight 4-bit weights into an int8
    """

    @property
    def compression_param_names(self) -> Tuple[str]:
        """
        Returns a tuple of compression parameter names introduced by
        the compressor during compression
        """
        return (
            "weight",
            "weight_scale",
            "weight_zero_point",
            "weight_g_idx",
        )

    def compression_param_info(
        self,
        weight_shape: torch.Size,
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> Dict[str, Tuple[torch.Size, torch.dtype]]:
        """
        Creates a dictionary of expected shapes and dtypes for each compression
            parameter used by the compressor

        :param weight_shape: uncompressed weight shape
        :param quantization_args: quantization parameters for the weight
        :return: dictionary mapping compressed parameter names to shape and dtype
        """
        dtype = quantization_args.pytorch_dtype()
        return {"weight": (weight_shape, dtype)}

    def compress_weight(
        self,
        weight: Tensor,
        scale: Tensor,
        quantization_args: QuantizationArgs,
        zero_point: Optional[Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compresses a single uncompressed weight

        :param weight: uncompressed weight tensor
        :param scale: quantization scale for weight
        :param quantization_args: quantization parameters for weight
        :param zero_point: quantization zero point for weight
        :param g_idx: optional mapping from column index to group index
        :param device: optional device to move compressed output to
        :return: dictionary of compressed weight data
        """
        compressed_dict = {}
        if can_quantize(weight, quantization_args):
            quantized_weight = quantize(
                x=weight,
                scale=scale,
                zero_point=zero_point,
                g_idx=g_idx,
                args=quantization_args,
                dtype=torch.int8,
            )
        else:
            quantized_weight = weight

        # for int4 pack to int8
        packed_weight = pack_interleave(quantized_weight)

        if device is not None:
            packed_weight = packed_weight.to(device)
        compressed_dict["weight_packed"] = packed_weight
        return compressed_dict


    def decompress_weight(
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> torch.Tensor:
        """
        Decompresses a single compressed weight

        :param compressed_data: dictionary of data needed for decompression
        :param quantization_args: quantization parameters for the weight
        :return: tensor of the decompressed weight
        """
        weight = compressed_data["weight_packed"]
        scale = compressed_data["weight_scale"]

        unpacked = unpack_interleave(weight)
        decompressed_weight = dequantize(
            x_q=unpacked, scale=scale
        )

        return decompressed_weight

