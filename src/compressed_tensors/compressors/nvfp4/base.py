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
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.utils import TensorStateDict, getattr_chain
from compressed_tensors.utils.impl_backend import ImplBackend


__all__ = ["NVFP4PackedCompressor", "compress_nvfp4"]


def _compress_nvfp4_meta_req(*args, **kwargs) -> bool:
    """Requirement for using the meta device NVFP4 compression backend."""
    state_dict = kwargs.get("state_dict")
    if state_dict is None:
        for arg in args:
            if isinstance(arg, dict):
                state_dict = arg
                break

    if state_dict is None:
        return False

    weight = state_dict.get("weight", None)
    return (
        weight is not None
        and hasattr(weight, "device")
        and weight.device.type == "meta"
    )


def _resolve_compress_args(*args, **kwargs):
    if len(args) == 3:
        return args[0], args[1], args[2]
    if len(args) == 2:
        if isinstance(args[0], type):
            return args[0], args[1], kwargs.get("scheme")
        return NVFP4PackedCompressor, args[0], args[1]
    cls = kwargs.get("cls", NVFP4PackedCompressor)
    state_dict = kwargs.get("state_dict")
    scheme = kwargs.get("scheme")
    if len(args) == 1:
        if isinstance(args[0], type):
            cls = args[0]
        elif isinstance(args[0], dict):
            state_dict = args[0]
    return cls, state_dict, scheme


@ImplBackend.register("compress_nvfp4", _compress_nvfp4_meta_req, 0)
def compress_nvfp4_meta(*args, **kwargs) -> TensorStateDict:
    """
    Construct meta tensors for weight_packed and weight_scale without computing FLOPs.
    """
    cls, state_dict, scheme = _resolve_compress_args(*args, **kwargs)
    state_dict = state_dict.copy()
    weight = state_dict.pop("weight")
    scale = state_dict.pop("weight_scale")
    m, n = weight.shape
    if n % 2 != 0:
        raise ValueError(
            "tensor must have an even number of columns for nvfp4 compression"
        )
    state_dict["weight_packed"] = torch.empty(
        m, n // 2, dtype=torch.uint8, device="meta"
    )
    state_dict["weight_scale"] = cls._compress_scale(scale, scheme.weights)
    state_dict = cls._remove_symmetric_zp(state_dict, scheme)
    return state_dict


@ImplBackend.entrypoint("compress_nvfp4")
def compress_nvfp4(*args, **kwargs) -> TensorStateDict:
    """
    Compress a per-module state dict using NVFP4 format.

    Quantizes the weight and packs into uint8 as ``weight_packed``.
    Compresses the scale according to ``scheme.weights.scale_dtype``.
    Removes the raw ``weight``.

    :param state_dict: local-name state dict (weight, weight_scale, …)
    :param scheme: quantization scheme for the weight
    :return: compressed state dict
    """
    cls, state_dict, scheme = _resolve_compress_args(*args, **kwargs)
    state_dict = state_dict.copy()
    weight = state_dict.pop("weight")
    scale = state_dict.pop("weight_scale")
    global_scale = state_dict.get("weight_global_scale", None)
    zero_point = state_dict.get("weight_zero_point", None)
    weights = scheme.weights

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


# Backwards compatibility alias for _skip_meta_device
_skip_meta_device = compress_nvfp4_meta
ImplBackend._fn_registry["_skip_meta_device"] = compress_nvfp4_meta


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
        if not getattr_chain(scheme, "input_activations.dynamic", True):
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
        return compress_nvfp4(cls, state_dict, scheme)

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
