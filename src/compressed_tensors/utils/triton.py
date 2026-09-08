# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Derived from https://github.com/vllm-project/vllm/blob/main/vllm/triton_utils/importing.py  # noqa: E501

import types

import torch
from loguru import logger


__all__ = [
    "HAS_TRITON",
    "triton",
    "tl",
    "tldevice",
    "gluon",
    "gl",
    "aggregate",
    "triton_req",
]


# Placeholders


def cdiv(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return -(-a // b)


class TritonPlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.__version__ = "3.4.0"
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")
        self.Config = self._dummy_decorator("Config")
        self.cdiv = cdiv
        self.language = TritonLanguagePlaceholder()

    def _dummy_decorator(self, name):
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

        return decorator


class TritonLanguagePlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = lambda x: x  # passthrough so tl.constexpr(n) == n
        self.dtype = None
        self.int64 = None
        self.int32 = None
        self.tensor = None
        self.exp = None
        self.exp2 = None
        self.log = None
        self.log2 = None


# Import and Export

HAS_TRITON = True

try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.language.core import _aggregate as aggregate

except ImportError:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()
    gluon = TritonLanguagePlaceholder()
    gl = TritonLanguagePlaceholder()
    aggregate = TritonLanguagePlaceholder()
    HAS_TRITON = False

except Exception as exception:
    logger.warning(
        f"Unexpected exception when loading triton, {exception}. "
        "Disabling Triton for now."
    )
    HAS_TRITON = False


# Backend Helpers


def triton_req(*args, **kwargs) -> bool:
    """Standard requirement for using triton, to be used with `ImplBackend`"""
    x: torch.Tensor = args[0]
    return HAS_TRITON and (x.is_cuda or x.is_xpu)
