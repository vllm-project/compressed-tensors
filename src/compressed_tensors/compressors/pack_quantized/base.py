# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
from compressed_tensors.compressors.base import (
    COMPRESSIBLE_MODULE_TYPES,
    BaseCompressor,
)
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.utils import TensorStateDict, getattr_chain


__all__ = ["PackedQuantizationCompressor"]


PACK_ZP_STRATS = [
    QuantizationStrategy.GROUP.value,
    QuantizationStrategy.CHANNEL.value,
]


@BaseCompressor.register(name=CompressionFormat.pack_quantized.value)
class PackedQuantizationCompressor(BaseCompressor):
    """
    Compresses a quantized weight by packing multiple sub-8-bit INT values into
    int32s using dense cross-element packing. Supports num_bits in [1, 8]; 32
    consecutive elements are packed into exactly num_bits int32 words with no
    wasted bits.
    """

    @classmethod
    def compression_param_names(cls, scheme: QuantizationScheme) -> tuple[str]:
        param_names = (
            "weight_packed",
            "weight_scale",
            "weight_shape",
        )
        if not getattr_chain(scheme, "weights.symmetric", True):
            param_names += ("weight_zero_point",)
        if getattr_chain(scheme, "weights.actorder", None) == ActivationOrdering.GROUP:
            param_names += ("weight_g_idx",)
        if (
            getattr_chain(scheme, "input_activations.strategy", None)
            == QuantizationStrategy.TENSOR_GROUP
        ):
            param_names += ("input_global_scale",)
        return param_names

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict.

        Quantizes the weight, packs it into int32 as ``weight_packed``, stores
        the original shape as ``weight_shape``, and removes ``weight``. If the
        quantization is asymmetric (GROUP or CHANNEL strategy) the zero-point is
        also packed; otherwise it is dropped.

        :param state_dict: local-name state dict (weight, weight_scale, …)
        :param quantization_args: quantization parameters for the weight
        :param device: device to move compressed tensors to
        :return: compressed state dict
        """
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.get("weight_scale")
        zero_point = state_dict.get("weight_zero_point", None)
        g_idx = state_dict.get("weight_g_idx", None)
        weights = scheme.weights

        if weight.device.type == "meta":
            packed_cols = math.ceil(weight.shape[-1] * weights.num_bits / 32)
            packed_shape = (*weight.shape[:-1], packed_cols)
            state_dict["weight_packed"] = torch.empty(
                packed_shape, dtype=torch.int32, device="meta"
            )
            state_dict["weight_shape"] = torch.tensor(weight.shape)
            state_dict = cls._remove_symmetric_zp(state_dict, scheme)
            return state_dict

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            zero_point=zero_point,
            g_idx=g_idx,
            args=scheme.weights,
            dtype=torch.int8,
        )
        state_dict["weight_packed"] = pack_to_int32(quantized_weight, weights.num_bits)
        state_dict["weight_shape"] = torch.tensor(weight.shape)

        if not weights.symmetric and weights.strategy in PACK_ZP_STRATS:
            assert zero_point is not None, "Asymmetric quant requires zero-point values"
            packed_zp = pack_to_int32(zero_point, weights.num_bits, packed_dim=0)
            state_dict["weight_zero_point"] = packed_zp.contiguous()

        state_dict = cls._remove_symmetric_zp(state_dict, scheme)

        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        Unpacks ``weight_packed`` back to the original weight, removes
        ``weight_packed``, and unpacks the zero-point if present.

        :param state_dict: local-name state dict (weight_packed, weight_scale, …)
        :param quantization_args: quantization parameters for the weight
        :return: decompressed state dict with weight in float dtype
        """
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")
        scale = state_dict.get("weight_scale")
        zero_point = state_dict.get("weight_zero_point", None)
        g_idx = state_dict.get("weight_g_idx", None)
        original_shape = state_dict.get("weight_shape")
        weights = scheme.weights

        if packed.device.type == "meta":
            state_dict["weight"] = torch.empty(
                tuple(original_shape.tolist()),
                dtype=scale.dtype,
                device="meta",
            )
            return state_dict

        # Unpack zero_point before dequantization if needed
        if not weights.symmetric and weights.strategy in PACK_ZP_STRATS:
            assert zero_point is not None, "Asymmetric quant requires zero-point values"
            original_zp_shape = (*original_shape[:-1], scale.shape[-1])
            zero_point = unpack_from_int32(
                zero_point, weights.num_bits, original_zp_shape, packed_dim=0
            )
            state_dict["weight_zero_point"] = zero_point

        unpacked = unpack_from_int32(packed, weights.num_bits, original_shape)
        state_dict["weight"] = dequantize(
            x_q=unpacked,
            scale=scale,
            zero_point=zero_point,
            g_idx=g_idx,
        )

        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """Pack quantized matches INT-only weight quantization with 1..8 bits.
        Excludes schemes with floating-point activation quantization."""
        if scheme.input_activations is not None:
            if scheme.input_activations.type == QuantizationType.FLOAT.value:
                return False
        return (
            module_type in COMPRESSIBLE_MODULE_TYPES
            and scheme.weights is not None
            and 1 <= scheme.weights.num_bits <= 8
            and scheme.weights.type == QuantizationType.INT.value
        )
