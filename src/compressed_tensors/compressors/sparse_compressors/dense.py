# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from torch import Tensor


if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationScheme


@BaseCompressor.register(name=CompressionFormat.dense.value)
class DenseCompressor(BaseCompressor):
    """
    Identity compressor for dense models, returns the original state_dict
    """

    @property
    def compression_param_names(self) -> tuple[str, ...]:
        """
        Returns a tuple of compression parameter names introduced by
        the compressor during compression
        """
        return ()

    def compress(self, model_state: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return model_state

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu", **kwargs
    ) -> Generator[tuple[str, Tensor], None, None]:
        return iter([])

    def decompress_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
    ) -> Generator[tuple[str, dict[str, Tensor]], None, None]:
        for key, value in state_dict.items():
            yield key, value

    def decompress_module_from_state_dict(
        self,
        prefix: str,
        state_dict: dict[str, torch.Tensor],
        scheme: "QuantizationScheme",
    ) -> dict[str, torch.Tensor]:
        """
        This function is implemented as a workaround because of how
        `ModelCompressor.quantization_compressor` can be set to either
        an instance of `BaseQuantizationCompressor` or `DenseCompressor`.
        """
        return state_dict.copy()
