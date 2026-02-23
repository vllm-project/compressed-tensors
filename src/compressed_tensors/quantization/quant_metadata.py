# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

from compressed_tensors.utils import disable_hf_hook
from torch.nn import Module


__all__ = ["QuantizationMetadata", "KVCacheScaleType"]


class KVCacheScaleType(Enum):
    KEY = "k_scale"
    VALUE = "v_scale"


class QuantizationMetadata:
    """
    Container class for metadata related to quantization
    """

    @staticmethod
    def all_qparam_names():
        """
        All quantization parameter names that might be registered
        onto a module during lifecycle (excluding serialized parameters)
        """
        return [KVCacheScaleType.KEY.value, KVCacheScaleType.VALUE.value] + [
            f"{base_name}_{suffix}"
            for base_name in ("input", "weight", "output")
            for suffix in (
                "global_scale",
                "scale",
                "zero_point",
                "g_idx",
            )
        ]

    @classmethod
    def clear_all_qparams(cls, module: Module):
        """
        Remove all parameters related to quantization that might have
        been registered onto a module previously in lifecycle (excluding
        serialized parameters)

        :param module: Module to clear
        """
        for key in cls.all_qparam_names():
            if hasattr(module, key):
                delattr(module, key)

    @classmethod
    def clear_quantization(cls, module: Module):
        """
        Remove all artifacts of quantization from module, non-recursively.
        Artifacts include any qparams, quantization_scheme, or wrapped
        forward method that might have been altered previously in lifecycle.

        `quantization_status` and `quantization_enabled` are left unchanged.

        :param module: Module to clear
        """
        with disable_hf_hook(module):
            # Unwrap forward call
            if hasattr(module.forward, "__wrapped__"):
                module.forward = module.forward.__wrapped__.__get__(module)

            # Clear any qparams
            cls.clear_all_qparams(module)

            # Clear quantization_scheme
            if hasattr(module, "quantization_scheme"):
                delattr(module, "quantization_scheme")
