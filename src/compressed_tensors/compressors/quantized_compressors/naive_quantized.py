# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.quantized_compressors.base import (
    BaseQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import (
    calculate_block_padding,
    can_quantize,
    pad_tensor_for_block_quant,
)
from torch import Tensor


__all__ = [
    "NaiveQuantizationCompressor",
    "IntQuantizationCompressor",
    "FloatQuantizationCompressor",
]


@BaseCompressor.register(name=CompressionFormat.naive_quantized.value)
class NaiveQuantizationCompressor(BaseQuantizationCompressor):
    """
    Implements naive compression for quantized models. Weight of each
    quantized layer is converted from its original float type to the closest Pytorch
    type to the type specified by the layer's QuantizationArgs.
    """

    @property
    def compression_param_names(self) -> tuple[str, ...]:
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
        quantization_args: QuantizationArgs | None = None,
    ) -> dict[str, tuple[torch.Size, torch.dtype]]:
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
        zero_point: Tensor | None = None,
        g_idx: torch.Tensor | None = None,
        device: torch.device | None = None,
        global_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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
        if global_scale is not None:
            raise ValueError(
                "global_scale is not supported for the NaiveQuantizationCompressor"
            )

        original_weight_shape = weight.shape

        # For block quantization, pad weight to divisible dimensions
        # This ensures proper scale alignment when layers are merged in vLLM
        if (
            quantization_args.strategy == QuantizationStrategy.BLOCK
            and quantization_args.block_structure is not None
        ):
            block_structure = tuple(quantization_args.block_structure)
            pad_rows, pad_cols = calculate_block_padding(weight.shape, block_structure)

            if pad_rows > 0 or pad_cols > 0:
                weight = pad_tensor_for_block_quant(weight, block_structure)

        if can_quantize(weight, quantization_args):
            quantized_weight = quantize(
                x=weight,
                scale=scale,
                zero_point=zero_point,
                g_idx=g_idx,
                args=quantization_args,
                dtype=quantization_args.pytorch_dtype(),
            )
        else:
            quantized_weight = weight

        if device is not None:
            quantized_weight = quantized_weight.to(device)

        if quantized_weight.shape != original_weight_shape:
            # return quantized_weight truncated back to original shape
            return {
                "weight": quantized_weight[
                    tuple([slice(k) for k in original_weight_shape])
                ]
            }
        return {"weight": quantized_weight}

    def decompress_weight(
        self,
        compressed_data: dict[str, Tensor],
        quantization_args: QuantizationArgs | None = None,
    ) -> torch.Tensor:
        """
        Decompresses a single compressed weight

        :param compressed_data: dictionary of data needed for decompression
        :param quantization_args: quantization parameters for the weight
        :return: tensor of the decompressed weight
        """
        weight = compressed_data["weight"]
        scale = compressed_data["weight_scale"]
        zero_point = compressed_data.get("weight_zero_point", None)
        g_idx = compressed_data.get("weight_g_idx", None)

        decompressed_weight = dequantize(
            x_q=weight, scale=scale, zero_point=zero_point, g_idx=g_idx
        )

        return decompressed_weight


@BaseCompressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(NaiveQuantizationCompressor):
    """
    Alias for integer quantized models
    """

    pass


@BaseCompressor.register(name=CompressionFormat.float_quantized.value)
class FloatQuantizationCompressor(NaiveQuantizationCompressor):
    """
    Alias for fp quantized models
    """

    pass
