# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import (
    COMPRESSIBLE_MODULE_TYPES,
    BaseCompressor,
)
from compressed_tensors.compressors.naive_quantized.fp8_block import (  # noqa: F401
    dequantize_fp8_block_weight,
    fp8_block_forward_emulation,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import (
    dequantize,
    forward_quantize,
    quantize,
)
from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant
from compressed_tensors.utils import TensorStateDict, getattr_chain
from compressed_tensors.utils.impl_backend import ImplBackend


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
    def compression_param_names(cls, scheme: QuantizationScheme) -> tuple[str]:
        param_names = (
            "weight",
            "weight_scale",
        )
        if not getattr_chain(scheme, "weights.symmetric", True):
            param_names += ("weight_zero_point",)
        return param_names

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
            args=weights,
            dtype=weights.pytorch_dtype(),
        )

        # Truncate back to original shape if padding was added
        if quantized_weight.shape != original_weight_shape:
            quantized_weight = quantized_weight[
                tuple([slice(v) for v in original_weight_shape])
            ]

        state_dict["weight"] = quantized_weight
        state_dict = cls._remove_symmetric_zp(state_dict, scheme)

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

        state_dict["weight"] = dequantize(
            x_q=weight,
            scale=scale,
            zero_point=zero_point,
            args=scheme.weights,
        )

        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """
        Naive quantization is the fallback compressor - it matches any quantized
        scheme that doesn't match a more specific compressor.
        """
        return module_type in COMPRESSIBLE_MODULE_TYPES and scheme.weights is not None


@BaseCompressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(NaiveQuantizationCompressor):
    """Alias for integer quantized models."""

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """Int quantized matches w8a8 int quantization."""
        return (
            module_type in COMPRESSIBLE_MODULE_TYPES
            and scheme.input_activations is not None
            and scheme.weights is not None
            and scheme.weights.type == QuantizationType.INT.value
        )


@BaseCompressor.register(name=CompressionFormat.float_quantized.value)
class FloatQuantizationCompressor(NaiveQuantizationCompressor):
    """Alias for fp quantized models."""

    @ImplBackend.entrypoint("fp8_block_forward")
    def fp8_block_compressed_forward(
        module: torch.nn.Linear, input: torch.Tensor
    ) -> torch.Tensor:
        """
        Run a linear forward pass directly on block-quantized FP8 weights.

        This is bound as the module's ``forward`` (``module`` plays the role of
        ``self``) by ``compress_module`` for BLOCK-strategy FP8 weights only, and
        serves as the eager fallback dispatched by ``ImplBackend`` when no
        accelerated backend applies. Activations are fake-quantized via
        ``forward_quantize`` and the weight is dequantized block-wise (via
        ``dequantize_fp8_block_weight``) before a dense ``F.linear``.

        The Triton ``fp8_block_forward_emulation`` backend (fused dequant matmul)
        takes priority on GPU.

        :param module: compressed linear module carrying ``weight``,
            ``weight_scale``, ``quantization_scheme`` and (optionally) ``bias``
        :param input: input activations of shape ``[*, in_features]``
        :return: output activations of shape ``[*, out_features]``
        """
        scheme: QuantizationScheme = module.quantization_scheme

        if scheme.input_activations is not None:
            input = forward_quantize(module, input, "input", scheme.input_activations)

        weight = dequantize_fp8_block_weight(
            module.weight,
            module.weight_scale,
            tuple(scheme.weights.block_structure),
            input.dtype,
        )
        return torch.nn.functional.linear(input, weight, getattr(module, "bias", None))

    # bound as the module's forward by ``compress_module`` (block strategy only,
    # see ``_binds_compressed_forward``)
    compressed_forward = fp8_block_compressed_forward

    @classmethod
    def _binds_compressed_forward(cls, module: torch.nn.Module) -> bool:
        """Only block-quantized FP8 has a specialized compressed forward; other
        FP8 strategies (tensor/channel/token) keep decompress-on-forward."""
        weights = getattr(module.quantization_scheme, "weights", None)
        return (
            super()._binds_compressed_forward(module)
            and weights is not None
            and weights.strategy == QuantizationStrategy.BLOCK
        )

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """Float quantized matches w8a8 float quantization."""
        return (
            module_type in COMPRESSIBLE_MODULE_TYPES
            and scheme.input_activations is not None
            and scheme.weights is not None
            and scheme.weights.type == QuantizationType.FLOAT.value
        )
