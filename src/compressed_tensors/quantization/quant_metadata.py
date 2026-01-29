from enum import Enum

from compressed_tensors.utils import delete_offload_parameter
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
                delete_offload_parameter(module, key)
