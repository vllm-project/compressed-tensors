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
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.compressed import (
    dequantize_weight,
    quantize_weight,
)
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    register_offload_parameter,
)


__all__ = ["pack_fp4_to_uint8", "unpack_fp4_from_uint8", "NVFP4PackedCompressor"]

FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


@BaseCompressor.register(name=CompressionFormat.nvfp4_pack_quantized.value)
class NVFP4PackedCompressor(BaseCompressor):
    """
    Implements compression of FP4 values. Weights of each quantized layer
    are packed into uint8. Only supports symmetric weight compression for now.
    """

    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        return (
            scheme.weights is not None
            and scheme.weights.type == QuantizationType.FLOAT
            and scheme.weights.num_bits == 4
            and scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP
            and scheme.weights.group_size == 16
            and scheme.weights.scale_dtype == torch.float8_e4m3fn
        )

    @classmethod
    def compress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        assert hasattr(module, "weight")
        assert hasattr(module, "weight_scale")

        with align_module_device(module):
            weight = quantize_weight(module, module.weight)
            weight = pack_fp4_to_uint8(weight)
            weight = torch.nn.Parameter(weight, requires_grad=False)
            delete_offload_parameter(module, "weight")
            register_offload_parameter(module, "weight_packed", weight)

            weight_scale = cls._compress_scale(module.weight_scale)
            weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
            register_offload_parameter(module, "weight_scale", weight_scale)

    @classmethod
    def decompress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        # original dtype information is destroyed during compression
        original_dtype = torch.bfloat16

        with align_module_device(module):
            weight_scale = module.weight_scale.to(original_dtype)
            weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
            register_offload_parameter(module, "weight_scale", weight_scale)

            weight = unpack_fp4_from_uint8(module.weight_packed, dtype=original_dtype)
            weight = dequantize_weight(module, weight)
            delete_offload_parameter(module, "weight_packed")
            register_offload_parameter(module, "weight", weight)

    @staticmethod
    def _compress_scale(scale: torch.Tensor) -> torch.Tensor:
        return scale.to(torch.float8_e4m3fn)

    @property
    def format(self) -> CompressionFormat:
        return CompressionFormat.nvfp4_pack_quantized


@BaseCompressor.register(name=CompressionFormat.mxfp4_pack_quantized.value)
class MXFP4PackedCompressor(NVFP4PackedCompressor):
    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        return (
            scheme.weights is not None
            and scheme.weights.type == QuantizationType.FLOAT
            and scheme.weights.num_bits == 4
            and scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP
            and scheme.weights.group_size == 32
            and scheme.weights.scale_dtype == torch.float8_e4m3fn
        )

    @staticmethod
    def _compress_scale(scale: torch.Tensor) -> torch.Tensor:
        scale_exp = 127 + torch.floor(torch.log2(scale)).to(torch.int32) - 2
        return scale_exp.to(torch.float8_e4m3fn)

    @classmethod
    def decompress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        raise NotImplementedError("MXFP4 Decompression is currently not supported")


@torch.compile(fullgraph=True, dynamic=True)
def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor with values in the fp4 range into uint8.
    As there are 16 valid fp4 values, two fp4 values can be
    packed into one uint8. Each fp4 value is mapped to its
    particular index (e.g. 0.5 is mapped to index 1, 6.0 is mapped
    to index 7) which is then represented using 4 bits. Consecutive
    pairs of 4 bits are then packed into an uint8.

    :param x: tensor to pack
    returns: a packed tensor in uint8
    """

    m, n = x.shape
    device = x.device

    if n % 2 != 0:
        raise ValueError(
            "tensor must have an even number of columns for nvfp4 compression"
        )

    # Create lookup table for FP4 values to indices
    # Map the absolute values to 0-7 indices
    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)

    # Find closest valid FP4 value index for each element
    abs_x = torch.abs(x)
    abs_diff_x = torch.abs(abs_x.unsqueeze(-1) - kE2M1)  # [m, n, 8]
    abs_indices = torch.argmin(abs_diff_x, dim=-1)  # [m, n]

    # Apply sign bit (bit 3) to get final 4-bit representation
    indices = abs_indices + (torch.signbit(x).to(torch.long) << 3)

    # Reshape to prepare for packing pairs of values
    indices = indices.reshape(-1)

    # Reshape to pair consecutive elements
    indices = indices.reshape(-1, 2)

    # Pack pairs of 4-bit values into 8-bit values
    packed = (indices[:, 0] | (indices[:, 1] << 4)).to(torch.uint8)

    return packed.reshape(m, n // 2)


kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


# reference: : https://github.com/vllm-project/vllm/pull/16362
@torch.compile(fullgraph=True, dynamic=True)
def unpack_fp4_from_uint8(
    a: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks uint8 values into fp4. Each uint8 consists of two fp4 values
    (i.e. first four bits correspond to one fp4 value, last four correspond to a
    consecutive fp4 value). The bits represent an index, which are mapped to an fp4
    value.

    :param a: tensor to unpack
    :param m: original dim 0 size of the unpacked tensor
    :param n: original dim 1 size of the unpacked tensor
    :param dtype: dense dtype to cast the unpacked tensor to
    """
    m, n_div_2 = a.shape[0]
    assert a.dtype == torch.uint8

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n_div_2 * 2).to(dtype=dtype)
