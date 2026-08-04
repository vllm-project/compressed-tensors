# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import (
    COMPRESSIBLE_MODULE_TYPES,
    BaseCompressor,
)
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.utils import getattr_chain


__all__ = ["MXFP4PackedCompressor"]


@BaseCompressor.register(name=CompressionFormat.mxfp4_pack_quantized.value)
class MXFP4PackedCompressor(NVFP4PackedCompressor):
    """
    Compressor for MXFP4 quantized models.

    Overrides scale compression to use log2 encoding (bias-127 exponent).
    """

    @classmethod
    def compression_param_names(cls, scheme: QuantizationScheme) -> tuple[str]:
        # MXFP4 uses GROUP strategy (not TENSOR_GROUP), so there is no
        # weight_global_scale parameter
        param_names = ("weight_packed", "weight_scale")
        if not getattr_chain(scheme, "weights.symmetric", True):
            param_names += ("weight_zero_point",)
        if not getattr_chain(scheme, "input_activations.dynamic", True):
            param_names += ("input_global_scale",)
        return param_names

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        scale_dtype = weights.scale_dtype or torch.uint8
        return compress_mx_scale(scale, scale_dtype)

    @classmethod
    def _decompress_scale(cls, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return decompress_mx_scale(scale).to(dtype)

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """MXFP4 matches FP4 with group_size=32."""
        return (
            module_type in COMPRESSIBLE_MODULE_TYPES
            and scheme.weights is not None
            and scheme.weights.num_bits == 4
            and scheme.weights.type == QuantizationType.FLOAT.value
            and scheme.weights.group_size == 32
        )
