# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.utils import TensorStateDict


__all__ = [
    "Int4PackedQuantizationCompressor",
    "pack_int4_values_to_int8",
    "unpack_int4_values_to_int8",
]


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
    low_nibbles = packed_tensor & 0x0F
    high_nibbles = (packed_tensor >> 4) & 0x0F

    out_shape = list(packed_tensor.shape)
    out_shape[-1] *= 2
    unpacked = torch.empty(out_shape, dtype=torch.int8, device=packed_tensor.device)

    unpacked[..., 0::2] = low_nibbles
    unpacked[..., 1::2] = high_nibbles
    return unpacked


@BaseCompressor.register(name=CompressionFormat.int4_quantized.value)
class Int4PackedQuantizationCompressor(BaseCompressor):
    """
    Compresses a w4a8 quantized model by packing every two int4 weights into one int8.

    Used for MoE models that quantize weights to int4 and activations to fp8.
    """

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict.

        Quantizes the weight to int8 values in the int4 range, then packs every
        two int4 values into a single int8 byte (low nibble first). Stores the
        result as ``weight_packed`` and removes the raw ``weight``.

        :param state_dict: local-name state dict (weight, weight_scale, ...)
        :param scheme: quantization scheme for the weight
        :return: compressed state dict
        """
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.get("weight_scale")
        zero_point = state_dict.get("weight_zero_point", None)
        g_idx = state_dict.get("weight_g_idx", None)
        weights = scheme.weights

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            zero_point=zero_point,
            g_idx=g_idx,
            args=weights,
            dtype=torch.int8,
        )

        state_dict["weight_packed"] = pack_int4_values_to_int8(
            quantized_weight.cpu()
        ).to(quantized_weight.device).contiguous()
        state_dict = cls._remove_symmetric_zp(state_dict, scheme)

        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        Unpacks ``weight_packed`` back to int4 values stored in int8 and
        dequantizes to float using the scale.

        :param state_dict: local-name state dict (weight_packed, weight_scale, ...)
        :param scheme: quantization scheme for the weight
        :return: decompressed state dict with weight in float dtype
        """
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")
        scale = state_dict.get("weight_scale")
        zero_point = state_dict.get("weight_zero_point", None)
        g_idx = state_dict.get("weight_g_idx", None)

        unpacked = unpack_int4_values_to_int8(packed)
        state_dict["weight"] = dequantize(
            x_q=unpacked,
            scale=scale,
            zero_point=zero_point,
            g_idx=g_idx,
        )

        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """Int4 packed matches w4a8 with int4 weights and fp8 input activations."""
        return (
            module_type == torch.nn.Linear
            and scheme.weights is not None
            and scheme.input_activations is not None
            and scheme.weights.num_bits == 4
            and scheme.weights.type == QuantizationType.INT.value
            and scheme.input_activations.num_bits == 8
            and scheme.input_activations.type == QuantizationType.FLOAT.value
        )
