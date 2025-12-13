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
import math
from typing import Literal, Union

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
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
    getattr_chain,
    register_offload_parameter,
)


__all__ = ["PackedQuantizationCompressor", "pack_to_int32", "unpack_from_int32"]


@BaseCompressor.register(name=CompressionFormat.pack_quantized.value)
class PackedQuantizationCompressor(BaseCompressor):
    """
    Compresses a quantized model by packing every eight 4-bit weights into an int32
    """

    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        return (
            scheme.input_activations is None
            and scheme.weights is not None
            and scheme.weights.type == QuantizationType.INT
            and scheme.weights.num_bits in (4, 8)
            and scheme.weights.strategy != QuantizationStrategy.TENSOR_GROUP
            and scheme.output_activations is None
        )

    @classmethod
    def compress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        assert hasattr(module, "weight")
        assert hasattr(module, "weight_scale")
        assert hasattr(module, "weight_zero_point")
        args: QuantizationArgs = getattr_chain(module, "quantization_scheme.weights")

        with align_module_device(module):
            weight_shape = torch.tensor(module.weight.shape)
            weight_shape = torch.nn.Parameter(weight_shape, requires_grad=False)
            register_offload_parameter(module, "weight_shape", weight_shape)

            weight = quantize_weight(module, module.weight)
            weight = pack_to_int32(weight, args.num_bits)
            weight = torch.nn.Parameter(weight, requires_grad=False)
            delete_offload_parameter(module, "weight")
            register_offload_parameter(module, "weight_packed", weight)

            if not args.symmetric and args.strategy in (
                QuantizationStrategy.GROUP.value,
                QuantizationStrategy.CHANNEL.value,
            ):
                zero_point = pack_to_int32(
                    module.weight_zero_point, args.num_bits, packed_dim=0
                )
                delete_offload_parameter(module, "weight_zero_point")
                register_offload_parameter(module, "weight_zero_point", zero_point)

    @classmethod
    def decompress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        assert hasattr(module, "weight_packed")
        assert hasattr(module, "weight_shape")
        assert hasattr(module, "weight_scale")
        args: QuantizationArgs = getattr_chain(module, "quantization_scheme.weights")

        with align_module_device(module):
            weight_shape = torch.Size(module.weight_shape)
            delete_offload_parameter(module, "weight_shape")

            weight = unpack_from_int32(
                module.weight_packed, args.num_bits, weight_shape
            )
            weight = dequantize_weight(module, weight)
            weight = torch.nn.Parameter(weight, requires_grad=False)
            delete_offload_parameter(module, "weight_packed")
            register_offload_parameter(module, "weight", weight)

            if not args.symmetric and args.strategy in (
                QuantizationStrategy.GROUP.value,
                QuantizationStrategy.CHANNEL.value,
            ):
                assert hasattr(module, "weight_zero_point")
                zero_point = pack_to_int32(
                    module.weight_zero_point, args.num_bits, packed_dim=0
                )
                delete_offload_parameter(module, "weight_zero_point")
                register_offload_parameter(module, "weight_zero_point", zero_point)

    @property
    def format(self) -> CompressionFormat:
        return CompressionFormat.pack_quantized


def pack_to_int32(
    value: torch.Tensor,
    num_bits: int,
    packed_dim: Union[Literal[0], Literal[1]] = 1,
) -> torch.Tensor:
    """
    Packs a tensor of quantized weights stored in int8 into int32s with padding

    Pseudocode:
     1. Shift wrt num_bits to convert to unsigned. num_bits=8
        [1,2] -> [129, 130]
     2. Pad to fill in 32 bits
        [129, 130] -> [129, 130, 0, 0]
     3. convert to binary align in order
        [129, 130, 0, 0] -> 00000000 00000000 10000010 10000001
     4. convert aligned binary to number
        00000000000000001000001010000001 -> 33409
     5. covert back to uint32
        33409 -> 33409

    :param value: tensor to pack
    :param num_bits: number of bits used to store underlying data, must be at least 1
    :returns: packed int32 tensor
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

    # Convert to unsigned range for packing, matching quantization offset
    offset = 1 << (num_bits - 1)
    value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    # Ensure contiguous memory for .view() operation
    value = value.contiguous()

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    # Use int32 here
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def unpack_from_int32(
    value: torch.Tensor,
    num_bits: int,
    shape: torch.Size,
    packed_dim: Union[Literal[0], Literal[1]] = 1,
) -> torch.Tensor:
    """
    Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
    original bit range.

    Return tensors in int8

    :param value: tensor to upack
    :param num_bits: number of bits to unpack each data point into
    :param shape: shape to unpack into, used to remove padding
    :returns: unpacked int8 tensor
    """
    if value.dtype is not torch.int32:
        raise ValueError(
            f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
        )

    if num_bits > 8:
        raise ValueError("Unpacking is only supported for less than 8 bits")

    pack_factor = 32 // num_bits

    # unpack
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked = torch.zeros(
            (value.shape[0], value.shape[1] * pack_factor),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[1])
        unpacked = unpacked[:, :original_row_size]
    else:
        unpacked = torch.zeros(
            (value.shape[0] * pack_factor, value.shape[1]),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[0])
        unpacked = unpacked[:original_row_size, :]

    # bits are packed in unsigned format, reformat to signed
    # update the value range from unsigned to signed
    offset = pow(2, num_bits) // 2
    unpacked = (unpacked - offset).to(torch.int8)

    return unpacked.contiguous()
