# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType
import torch
from compressed_tensors.compressors.base import (
    COMPRESSIBLE_MODULE_TYPES,
    BaseCompressor,
)
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
from compressed_tensors.compressors.mxfp4.linear import (  # noqa: F401
    dequantize_mxfp4_weight,
    mxfp4_forward_emulation,
    mxfp4_forward_fp4,
)
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.impl_backend import ImplBackend


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

    @ImplBackend.entrypoint("mxfp4_forward")
    def compressed_forward(module: torch.nn.Linear, input: torch.Tensor) -> MethodType:
        """
        Run a linear forward pass directly on the compressed weights.

        This is bound as the module's ``forward`` (``module`` plays the role of
        ``self``) and serves as the eager fallback, dispatched by ``ImplBackend``
        only when no accelerated backend applies. It emulates MXFP4 linear:
        activations are fake-quantized via ``forward_quantize`` and the weight is
        unpacked and dequantized (via ``decompress_mx_scale``) before a dense
        ``F.linear`` in full precision.

        Faster backends registered under ``"mxfp4_forward"`` take priority:

        - ``fp4`` (Blackwell): real MXFP4 matmul on FP4 tensor cores
        - ``emulation`` (Triton GPU): fused dequant-and-matmul Triton kernel

        :param module: compressed linear module carrying ``weight_packed``,
            ``weight_scale``, ``quantization_scheme`` and (optionally) ``bias``
        :param input: input activations of shape ``[*, in_features]``
        :return: output activations of shape ``[*, out_features]``
        """
        scheme: QuantizationScheme = module.quantization_scheme

        if scheme.input_activations is not None:
            input = forward_quantize(module, input, "input", scheme.input_activations)

        weight = dequantize_mxfp4_weight(
            module.weight_packed, module.weight_scale, input.dtype
        )
        return torch.nn.functional.linear(input, weight, getattr(module, "bias", None))

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
