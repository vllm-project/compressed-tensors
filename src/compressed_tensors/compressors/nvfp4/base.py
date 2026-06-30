# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import (
    COMPRESSIBLE_MODULE_TYPES,
    BaseCompressor,
)
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.quant_args import round_to_quantized_type_args
from compressed_tensors.utils import TensorStateDict, getattr_chain


__all__ = ["NVFP4PackedCompressor"]


@BaseCompressor.register(name=CompressionFormat.nvfp4_pack_quantized.value)
class NVFP4PackedCompressor(BaseCompressor):
    """
    Compressor for FP4 quantized models.

    Weights of each quantized layer are packed into uint8. Only supports
    symmetric weight compression.
    """

    @classmethod
    def compression_param_names(cls, scheme: QuantizationScheme) -> tuple[str]:
        param_names = (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
        )
        if not getattr_chain(scheme, "weights.symmetric", True):
            param_names += ("weight_zero_point",)
        if (
            getattr_chain(scheme, "input_activations.strategy", None)
            == QuantizationStrategy.TENSOR_GROUP
        ):
            param_names += ("input_global_scale",)
        return param_names

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        scale_dtype = weights.scale_dtype or torch.float8_e4m3fn
        return scale.to(scale_dtype)

    @classmethod
    def _decompress_scale(cls, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return scale.to(dtype)

    @classmethod
    def _adjust_scale_for_four_over_six(
        cls,
        weight: torch.Tensor,
        scale: torch.Tensor,
        global_scale: torch.Tensor | None,
        weights: QuantizationArgs,
    ) -> torch.Tensor:
        """
        Pre-adjust per-group scales for Four Over Six: for each group, compare
        MSE of standard quantization (scale-to-6) vs alternative (scale-to-4,
        i.e. scale * 1.5). Groups where scale-to-4 wins get their scale
        multiplied by 1.5 so that the stored scale reflects the actual
        quantization used.
        """
        from compressed_tensors.quantization.utils import calculate_range

        q_min, q_max = calculate_range(weights, weight.device)

        group_size = weights.group_size
        rows, cols = weight.shape
        num_groups = cols // group_size
        w_grouped = weight.reshape(rows, num_groups, group_size)

        if global_scale is not None:
            eff_scale = scale / global_scale
        else:
            eff_scale = scale.clone()

        eff_scale_3d = eff_scale.unsqueeze(-1)

        scaled_a = w_grouped / eff_scale_3d
        q_a = round_to_quantized_type_args(
            tensor=scaled_a, args=weights, min=q_min, max=q_max
        )
        dq_a = q_a.to(eff_scale.dtype) * eff_scale_3d

        eff_scale_b = eff_scale * 1.5
        eff_scale_b_3d = eff_scale_b.unsqueeze(-1)
        scaled_b = w_grouped / eff_scale_b_3d
        q_b = round_to_quantized_type_args(
            tensor=scaled_b, args=weights, min=q_min, max=q_max
        )
        dq_b = q_b.to(eff_scale_b.dtype) * eff_scale_b_3d

        mse_a = ((w_grouped - dq_a) ** 2).mean(dim=-1)
        mse_b = ((w_grouped - dq_b) ** 2).mean(dim=-1)

        use_4 = mse_b < mse_a
        adjusted_scale = scale.clone()
        adjusted_scale[use_4] = adjusted_scale[use_4] * 1.5
        return adjusted_scale.to(scale.dtype)

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict.

        Quantizes the weight and packs into uint8 as ``weight_packed``.
        Compresses the scale according to ``scheme.weights.scale_dtype``.
        Removes the raw ``weight``.

        :param state_dict: local-name state dict (weight, weight_scale, …)
        :param scheme: quantization scheme for the weight
        :return: compressed state dict
        """
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.pop("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)
        zero_point = state_dict.get("weight_zero_point", None)
        weights = scheme.weights

        if getattr(weights, "four_over_six", False):
            scale = cls._adjust_scale_for_four_over_six(
                weight, scale, global_scale, weights
            )

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            global_scale=global_scale,
            zero_point=zero_point,
            args=weights,
        )
        state_dict["weight_packed"] = pack_fp4_to_uint8(quantized_weight)
        state_dict["weight_scale"] = cls._compress_scale(scale, weights)
        state_dict = cls._remove_symmetric_zp(state_dict, scheme)

        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        Unpacks ``weight_packed`` back to fp4 values and dequantizes.
        Converts ``weight_scale`` back to float for dequantization.

        :param state_dict: local-name state dict (weight_packed, weight_scale, …)
        :param scheme: quantization scheme for the weight
        :return: decompressed state dict with weight in float dtype
        """
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")
        scale = state_dict.get("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)

        m, n = packed.shape
        unpacked = unpack_fp4_from_uint8(packed, m, n * 2)

        scale_float = cls._decompress_scale(scale, unpacked.dtype)

        state_dict["weight"] = dequantize(
            x_q=unpacked,
            scale=scale_float,
            global_scale=global_scale,
            dtype=unpacked.dtype,
        )
        state_dict["weight_scale"] = torch.nn.Parameter(
            scale_float, requires_grad=False
        )

        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """NVFP4 matches FP4 with group_size != 32 (or None)."""
        return (
            module_type in COMPRESSIBLE_MODULE_TYPES
            and scheme.weights is not None
            and scheme.weights.num_bits == 4
            and scheme.weights.type == QuantizationType.FLOAT.value
            and scheme.weights.group_size == 16
        )
