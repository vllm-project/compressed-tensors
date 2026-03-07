# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant
from compressed_tensors.utils import TensorStateDict


__all__ = [
    "NaiveQuantizationCompressor",
    "IntQuantizationCompressor",
    "FloatQuantizationCompressor",
]


@BaseCompressor.register(name=CompressionFormat.naive_quantized.value)
class NaiveQuantizationCompressor(BaseCompressor):
    """
    Naive quantization compressor.

    Each quantized layer's weight is converted from its original float dtype to
    the closest PyTorch dtype for the bit-width specified by QuantizationArgs.
    """

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict.

        Quantizes the weight to the dtype specified by the scheme's
        QuantizationArgs. Handles block quantization padding if needed.

        :param state_dict: local-name state dict (weight, weight_scale, …)
        :param scheme: quantization scheme for the weight
        :return: compressed state dict
        """
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.get("weight_scale")
        zero_point = state_dict.get("weight_zero_point", None)
        g_idx = state_dict.get("weight_g_idx", None)
        weights = scheme.weights

        original_weight_shape = weight.shape

        # For block quantization, pad weight to divisible dimensions
        if (
            weights.strategy == QuantizationStrategy.BLOCK
            and weights.block_structure is not None
        ):
            block_structure = tuple(weights.block_structure)
            weight = maybe_pad_tensor_for_block_quant(weight, block_structure)

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            zero_point=zero_point,
            g_idx=g_idx,
            args=weights,
            dtype=weights.pytorch_dtype(),
        )

        # Truncate back to original shape if padding was added
        if quantized_weight.shape != original_weight_shape:
            quantized_weight = quantized_weight[
                tuple([slice(v) for v in original_weight_shape])
            ]

        state_dict["weight"] = quantized_weight

        if weights.symmetric:
            state_dict.pop("weight_zero_point", None)

        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        Dequantizes the weight back to float dtype using the scale and
        zero-point from the state dict.

        :param state_dict: local-name state dict (weight, weight_scale, …)
        :param scheme: quantization scheme for the weight
        :return: decompressed state dict with weight in float dtype
        """
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.get("weight_scale")
        zero_point = state_dict.get("weight_zero_point", None)
        g_idx = state_dict.get("weight_g_idx", None)

        state_dict["weight"] = dequantize(
            x_q=weight,
            scale=scale,
            zero_point=zero_point,
            g_idx=g_idx,
        )

        return state_dict

    @classmethod
    def match(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """
        Naive quantization is the fallback compressor - it matches any quantized
        scheme that doesn't match a more specific compressor.
        """
        return module_type == torch.nn.Linear and scheme.weights is not None


@BaseCompressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(NaiveQuantizationCompressor):
    """Alias for integer quantized models."""

    @classmethod
    def match(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """Int quantized matches w8a8 int quantization."""
        return (
            module_type == torch.nn.Linear
            and scheme.input_activations is not None
            and scheme.weights is not None
            and scheme.weights.type == QuantizationType.INT.value
        )


@BaseCompressor.register(name=CompressionFormat.float_quantized.value)
class FloatQuantizationCompressor(NaiveQuantizationCompressor):
    """Alias for fp quantized models."""

    @classmethod
    def match(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """Float quantized matches w8a8 float quantization."""
        return (
            module_type == torch.nn.Linear
            and scheme.input_activations is not None
            and scheme.weights is not None
            and scheme.weights.type == QuantizationType.FLOAT.value
        )
