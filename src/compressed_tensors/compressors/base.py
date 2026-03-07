# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC

import torch
from compressed_tensors.compressors.format import get_module_format
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.registry import RegistryMixin
from compressed_tensors.utils import (
    TensorStateDict,
    get_direct_state_dict,
    replace_direct_state_dict,
)


__all__ = ["BaseCompressor", "compress_module", "decompress_module"]


class BaseCompressor(RegistryMixin, ABC):
    """
    Base class representing a model compression algorithm.

    New quantization compressors (dense, naive_quantized, pack_quantized, nvfp4,
    mxfp4) use the classmethod interface — they are never instantiated. Look up
    via BaseCompressor.get_value_from_registry(format) and call compress/decompress
    directly on the returned class.

    Legacy sparse compressors (sparse_bitmask, sparse_24_bitmask, marlin_24) still
    use the instance-based interface and are instantiated via load_from_registry.
    """

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict.

        Keys are *local* names (``weight``, ``weight_scale``, …), not prefixed
        with the module path.

        :param state_dict: per-module state dict with local parameter names
        :param scheme: quantization scheme containing quantization parameters
        :return: compressed per-module state dict
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement the classmethod compress interface"
        )

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        Keys are *local* names (``weight_packed``, ``weight_scale``, …).

        :param state_dict: compressed per-module state dict with local parameter names
        :param scheme: quantization scheme containing quantization parameters
        :return: decompressed per-module state dict
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement the classmethod decompress interface"
        )

    @classmethod
    def compress_module(cls, module: torch.nn.Module) -> None:
        """
        Compress a module in-place by compressing its state dict.

        Extracts the module's parameters and buffers, compresses them using the
        compress classmethod, and replaces the module's state with the compressed
        version.

        :param module: the module to compress in-place
        """
        scheme = getattr(module, "quantization_scheme")
        state_dict = get_direct_state_dict(module)
        compressed_state_dict = cls.compress(state_dict, scheme)
        del state_dict
        replace_direct_state_dict(module, compressed_state_dict)

    @classmethod
    def decompress_module(cls, module: torch.nn.Module) -> None:
        """
        Decompress a module in-place by decompressing its state dict.

        Extracts the module's parameters and buffers, decompresses them using the
        decompress classmethod, and replaces the module's state with the decompressed
        version.

        :param module: the module to decompress in-place
        """
        scheme = getattr(module, "quantization_scheme")
        state_dict = get_direct_state_dict(module)
        decompressed_state_dict = cls.decompress(state_dict, scheme)
        del state_dict
        replace_direct_state_dict(module, decompressed_state_dict)

    @classmethod
    def match(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """
        Determine if this compressor is applicable for the given module type and scheme.

        Examines the module type and quantization scheme and determines whether this
        compressor can handle the module's compression requirements.

        :param module_type: the type of the module to check for compatibility
        :param scheme: the quantization scheme to check for compatibility
        :return: True if this compressor can handle the module, False otherwise
        """
        raise NotImplementedError(f"{cls.__name__} does not implement match")


def compress_module(module: torch.nn.Module):
    """
    Compress a module which has had quantization applied to it

    :param module: module to compress inplace
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not isinstance(scheme, QuantizationScheme):
        return

    if scheme.format is None:
        scheme.format = get_module_format(type(module), scheme)

    compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
    compressor.compress_module(module)


def decompress_module(module: torch.nn.Module):
    """
    Decompress a module which has had quantization applied to it

    :param module: module to decompress inplace
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not isinstance(scheme, QuantizationScheme):
        return

    if scheme.format is None:
        scheme.format = get_module_format(type(module), scheme)

    compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
    compressor.decompress_module(module)
