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

from typing import Dict, Optional, Tuple

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.quantized_compressors.base import (
    BaseQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import (
    can_quantize,
    calculate_block_padding,
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
        if global_scale is not None:
            raise ValueError(
                "global_scale is not supported for the NaiveQuantizationCompressor"
            )

        result = {}
        original_shape = None

        # For block quantization, pad weight to divisible dimensions
        # This ensures proper scale alignment when layers are merged in vLLM
        if (
            quantization_args.strategy == QuantizationStrategy.BLOCK
            and quantization_args.block_structure is not None
        ):
            block_structure = tuple(quantization_args.block_structure)
            pad_rows, pad_cols = calculate_block_padding(weight.shape, block_structure)

            if pad_rows > 0 or pad_cols > 0:
                original_shape = weight.shape
                weight, _ = pad_tensor_for_block_quant(weight, block_structure)

                # Also pad the scale tensor to match the padded weight dimensions
                # Scale shape is (num_row_blocks, num_col_blocks)
                padded_rows, padded_cols = weight.shape[-2], weight.shape[-1]
                block_height, block_width = block_structure
                expected_scale_rows = padded_rows // block_height
                expected_scale_cols = padded_cols // block_width

                if scale.shape[0] < expected_scale_rows:
                    scale_pad_rows = expected_scale_rows - scale.shape[0]
                    scale = torch.nn.functional.pad(
                        scale, (0, 0, 0, scale_pad_rows), mode="constant", value=0
                    )
                if scale.shape[1] < expected_scale_cols:
                    scale_pad_cols = expected_scale_cols - scale.shape[1]
                    scale = torch.nn.functional.pad(
                        scale, (0, scale_pad_cols, 0, 0), mode="constant", value=0
                    )

                # Pad zero_point if present
                if zero_point is not None:
                    if zero_point.shape[0] < expected_scale_rows:
                        zp_pad_rows = expected_scale_rows - zero_point.shape[0]
                        zero_point = torch.nn.functional.pad(
                            zero_point,
                            (0, 0, 0, zp_pad_rows),
                            mode="constant",
                            value=0,
                        )
                    if zero_point.shape[1] < expected_scale_cols:
                        zp_pad_cols = expected_scale_cols - zero_point.shape[1]
                        zero_point = torch.nn.functional.pad(
                            zero_point,
                            (0, zp_pad_cols, 0, 0),
                            mode="constant",
                            value=0,
                        )

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

        result["weight"] = quantized_weight

        # If weight was padded for block quantization, return the padded scale
        # Note: We don't save weight_shape_original to avoid issues with vLLM weight loading.
        # The config.json should be updated with padded dimensions by the quantization tool.
        # Don't add zero_point here - base class _skip_zp() will omit it for symmetric quant
        if original_shape is not None:
            result["weight_scale"] = scale

        return result

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
        weight = compressed_data["weight"]
        scale = compressed_data["weight_scale"]
        zero_point = compressed_data.get("weight_zero_point", None)
        g_idx = compressed_data.get("weight_g_idx", None)

        decompressed_weight = dequantize(
            x_q=weight, scale=scale, zero_point=zero_point, g_idx=g_idx
        )

        # Note: For block-quantized models with padding, the decompressed weight
        # will remain padded. The config.json should reflect the padded dimensions.

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
