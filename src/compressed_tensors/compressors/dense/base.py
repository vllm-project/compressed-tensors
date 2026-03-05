# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils import TensorStateDict


__all__ = ["DenseCompressor"]


@BaseCompressor.register(name=CompressionFormat.dense.value)
class DenseCompressor(BaseCompressor):
    """
    Identity compressor for dense models — both compress and decompress
    return the state dict unchanged.
    """

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict.

        For dense models, this is a no-op.

        :param state_dict: local-name state dict (weight, ...)
        :param scheme: quantization scheme (unused)
        :return: unmodified state dict
        """
        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        For dense models, this is a no-op.

        :param state_dict: local-name state dict (weight, ...)
        :param scheme: quantization scheme (unused)
        :return: unmodified state dict
        """
        return state_dict

    @classmethod
    def match(cls, module: torch.nn.Module) -> bool:
        """Dense compressor matches when there's no weight quantization."""
        return True
