# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC
from itertools import chain
from typing import Tuple

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.registry import RegistryMixin
from compressed_tensors.utils import TensorStateDict
from compressed_tensors.utils.helpers import getattr_chain


__all__ = ["BaseCompressor"]


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
    def compress_module(cls, module: torch.nn.Module):
        """
        Compress a module in-place by compressing its state dict.

        Extracts the module's parameters and buffers, compresses them using the
        compress classmethod, and replaces the module's state with the compressed
        version.

        :param module: the module to compress in-place
        """
        scheme = getattr(module, "quantization_scheme")
        state_dict = _get_direct_state_dict(module)
        compressed_state_dict = cls.compress(state_dict, scheme)
        del state_dict
        _replace_direct_state_dict(module, compressed_state_dict)

    @classmethod
    def decompress_module(cls, module: torch.nn.Module):
        """
        Decompress a module in-place by decompressing its state dict.

        Extracts the module's parameters and buffers, decompresses them using the
        decompress classmethod, and replaces the module's state with the decompressed
        version.

        :param module: the module to decompress in-place
        """
        scheme = getattr(module, "quantization_scheme")
        state_dict = _get_direct_state_dict(module)
        decompressed_state_dict = cls.decompress(state_dict, scheme)
        del state_dict
        _replace_direct_state_dict(module, decompressed_state_dict)

    @classmethod
    def match(cls, module: torch.nn.Module) -> bool:
        """
        Determine if this compressor is applicable for the given module.

        Examines the module's quantization scheme and determines whether this
        compressor can handle the module's compression requirements.

        :param module: the module to check for compatibility
        :return: True if this compressor can handle the module, False otherwise
        """
        raise NotImplementedError(f"{cls.__name__} does not implement match")

    @classmethod
    def _unpack_quantization(
        cls, module: torch.nn.Module
    ) -> Tuple[type, QuantizationArgs | None, QuantizationArgs | None]:
        """
        Extract quantization information from a module.

        Helper method to retrieve the module type and its quantization parameters
        for input activations and weights.

        :param module: the module to extract quantization info from
        :return: tuple of (module_type, input_quantization, weight_quantization)
        """
        return (
            type(module),
            getattr_chain(module, "quantization_scheme.input_activations", None),
            getattr_chain(module, "quantization_scheme.weights", None),
        )


def _get_direct_state_dict(module: torch.nn.Module) -> TensorStateDict:
    """
    Extract a state dict directly from a module's parameters and buffers.

    Returns tensor data (unwrapped from Parameter/Buffer wrappers) for all
    parameters and buffers in the module. Does not recurse into child modules.

    :param module: the module to extract state from
    :return: dict mapping parameter/buffer names to their tensor data
    """
    return {
        name: (
            tensor.data
            if isinstance(tensor, (torch.nn.Parameter, torch.nn.Buffer))
            else tensor
        )
        for name, tensor in chain(module._parameters.items(), module._buffers.items())
    }


def _replace_direct_state_dict(
    module: torch.nn.Module, new_state_dict: TensorStateDict
):
    """
    Replace a module's parameters and buffers with a new state dict.

    Removes parameters/buffers that exist in the old state but not the new state,
    and adds/updates parameters from the new state dict. All new tensors are
    added as non-trainable parameters (not buffers). Skips unchanged values
    for efficiency.

    :param module: the module to update
    :param new_state_dict: dict of new parameter/buffer values
    """
    old_state_dict = _get_direct_state_dict(module)

    for name, old_value in old_state_dict.items():
        # remove attributes that don't exist in the new state
        if name not in new_state_dict:
            delattr(module, name)

    for name, new_value in new_state_dict.items():
        # skip unchanged values
        if name not in old_state_dict or old_state_dict[name] is not new_value:
            # overwrite (not update) if param already existed
            if hasattr(module, name):
                delattr(module, name)

            # treat all new tensors as parameters (not buffers)
            setattr(module, name, torch.nn.Parameter(new_value, requires_grad=False))
