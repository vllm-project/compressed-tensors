"""
Miscelaneous helpers for the quantization lifecycle
"""

from torch.nn import Module


__all__ = [
    "enable_quantization",
    "disable_quantization",
]


def enable_quantization(module: Module):
    module.quantization_enabled = True


def disable_quantization(module: Module):
    module.quantization_enabled = False
