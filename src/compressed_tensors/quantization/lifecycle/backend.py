# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pluggable backends for the elementwise quantization lifecycle ops.

The lifecycle leaf ops (``quantize``, ``dequantize``, ``quantize_dequantize``)
run on every weight/activation group during calibration and are called
hundreds of thousands of times for even small models (see #766). Routing them
through a backend registry keeps the reference ``eager`` implementation as the
default while letting accelerated backends (``torch.compile``, hand written
triton kernels, ...) register under a name and slot in without touching call
sites.

The active backend is selected globally via :func:`set_quantization_backend`
or the ``COMPRESSED_TENSORS_QUANT_BACKEND`` environment variable and defaults
to ``eager``. A backend that is not usable for a given tensor (per
:meth:`QuantizationBackend.is_available`) transparently falls back to eager.
"""

import os

import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    round_to_quantized_type_args,
)
from compressed_tensors.registry import RegistryMixin


__all__ = [
    "QuantizationBackend",
    "EagerQuantizationBackend",
    "get_quantization_backend",
    "set_quantization_backend",
]

_DEFAULT_BACKEND = "eager"
_ACTIVE_BACKEND = os.environ.get("COMPRESSED_TENSORS_QUANT_BACKEND", _DEFAULT_BACKEND)


class QuantizationBackend(RegistryMixin):
    """Base class for quantization lifecycle backends.

    Subclasses register with :meth:`register` and implement the three leaf ops
    as static methods. The signatures mirror the historical private helpers so
    existing call sites are unchanged.
    """

    registry_requires_subclass = True

    @staticmethod
    def is_available(x: torch.Tensor, args: QuantizationArgs) -> bool:
        """Whether this backend can handle the given tensor/args. Backends that
        return False for a given input are skipped in favor of eager."""
        return True

    @staticmethod
    def quantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        args: QuantizationArgs,
        dtype: torch.dtype | None = None,
        global_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def dequantize(
        x_q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
        global_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def quantize_dequantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        args: QuantizationArgs,
        global_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


@QuantizationBackend.register(name="eager")
class EagerQuantizationBackend(QuantizationBackend):
    """Reference PyTorch implementation. Bit identical to the pre-dispatch code."""

    @staticmethod
    def quantize(
        x, scale, zero_point, q_min, q_max, args, dtype=None, global_scale=None
    ):
        # if a global scale is optionally provided, use it
        # to further scale the local `scale` parameter
        if global_scale is not None:
            scale = scale / global_scale

        scaled = x / scale

        if zero_point is not None:
            scaled += zero_point.to(x.dtype)

        # clamp and round
        quantized_value = round_to_quantized_type_args(
            tensor=scaled, args=args, min=q_min, max=q_max
        )

        if dtype is not None:
            quantized_value = quantized_value.to(dtype)

        return quantized_value

    @staticmethod
    def dequantize(x_q, scale, zero_point=None, global_scale=None):
        # if a global scale is optionally provided, use it
        # to further scale the local `scale` parameter
        if global_scale is not None:
            scale = scale / global_scale

        dequant_value = x_q.to(scale.dtype)

        if zero_point is not None:
            dequant_value = dequant_value - zero_point.to(scale.dtype)

        return dequant_value * scale

    @staticmethod
    def quantize_dequantize(
        x, scale, zero_point, q_min, q_max, args, global_scale=None
    ):
        # compute effective scale once
        if global_scale is not None:
            scale = scale / global_scale

        scaled = x / scale

        if zero_point is not None:
            scaled += zero_point.to(x.dtype)

        # clamp and round (stays in float — no int8/fp8 intermediate)
        quantized = round_to_quantized_type_args(
            tensor=scaled, args=args, min=q_min, max=q_max
        )

        # dequantize: subtract zero_point and multiply by scale
        # cast to scale.dtype to match _dequantize behavior
        dequant = quantized.to(scale.dtype)
        if zero_point is not None:
            dequant = dequant - zero_point.to(scale.dtype)

        return dequant * scale


def set_quantization_backend(name: str) -> None:
    """Set the active quantization backend by registered name.

    :param name: registered backend name (e.g. ``"eager"``). Raises if unknown.
    """
    QuantizationBackend.get_value_from_registry(name)  # validate existence
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = name


def get_quantization_backend(
    x: torch.Tensor | None = None, args: QuantizationArgs | None = None
) -> type[QuantizationBackend]:
    """Return the active backend, falling back to eager when the active backend
    is not available for the given tensor/args."""
    backend = QuantizationBackend.get_value_from_registry(_ACTIVE_BACKEND)
    if _ACTIVE_BACKEND != _DEFAULT_BACKEND and x is not None:
        if not backend.is_available(x, args):
            return QuantizationBackend.get_value_from_registry(_DEFAULT_BACKEND)
    return backend
