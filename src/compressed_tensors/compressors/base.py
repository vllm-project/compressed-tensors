# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC
from typing import Optional

import torch
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme, QuantizationStatus
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
        Compress a per-module state dict. Does not modify the input.

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
        Decompress a per-module state dict. Does not modify the input.

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
        replace_direct_state_dict(module, compressed_state_dict)

        module.quantization_status = QuantizationStatus.COMPRESSED

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
        replace_direct_state_dict(module, decompressed_state_dict)

        module.quantization_status = QuantizationStatus.DECOMPRESSED

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

    @classmethod
    def _remove_symmetric_zp(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Remove zero points where quantization arguments do not specify asymmetric quant.
        This is required during compression because vLLM does not support loading
        zero point parameters for symmetric schemes.
        TODO: remove zero points from initialization

        :param state_dict: state dict containing extra zero point parameters
        :return: state dict with extra zero point parameters removed
        """
        if scheme.input_activations and scheme.input_activations.symmetric:
            state_dict.pop("input_zero_point", None)
        if scheme.weights and scheme.weights.symmetric:
            state_dict.pop("weight_zero_point", None)
        if scheme.output_activations and scheme.output_activations.symmetric:
            state_dict.pop("output_zero_point", None)

        return state_dict


def compress_module(
    module: torch.nn.Module, format: Optional[CompressionFormat] = None
):
    """
    Compress a module which has had quantization applied to it. Sets the
    module's quantization format attribute to whichever format was used to compress.

    The format used to compress will be found from one of the following locations:
    1. the `format` argument passed to this function
    2. the attached `quantization_scheme.format` attribute
    3. format inferred from `infer_module_format`

    :param module: module to compress inplace
    :param format: force override for compression format
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not isinstance(scheme, QuantizationScheme):
        return

    scheme.format = CompressionFormat(
        format or scheme.format or infer_module_format(type(module), scheme)
    )
    compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
    compressor.compress_module(module)


def decompress_module(
    module: torch.nn.Module, format: Optional[CompressionFormat] = None
):
    """
    Decompress a module which has had quantization applied to it. Sets the
    module's quantization format attribute to whichever format was used to decompress.

    The format used to decompress will be found from one of the following locations:
    1. the `format` argument passed to this function
    2. the attached `quantization_scheme.format` attribute
    3. format inferred from `infer_module_format`

    :param module: module to decompress inplace
    :param format: force override for decompression format
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not isinstance(scheme, QuantizationScheme):
        return

    scheme.format = CompressionFormat(
        format or scheme.format or infer_module_format(type(module), scheme)
    )
    compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
    compressor.decompress_module(module)
