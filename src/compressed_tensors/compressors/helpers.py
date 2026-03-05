# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from pathlib import Path

import torch
from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.utils import deprecated
from torch import Tensor


__all__ = [
    "load_compressed",
    "save_compressed",
    "save_compressed_model",
]


@deprecated("LLM Compressor's `model_free_ptq` pathway")
def save_compressed(
    tensors: dict[str, Tensor],
    save_path: str | Path,
    compression_format: CompressionFormat | None = None,
):
    raise NotImplementedError()


@deprecated("LLM Compressor's `model_free_ptq` pathway")
def load_compressed(
    compressed_tensors: str | Path,
    compression_config: SparsityCompressionConfig = None,
    device: str | None = "cpu",
) -> Generator[tuple[str, Tensor], None, None]:
    raise NotImplementedError()


@deprecated("LLM Compressor's `model_free_ptq` pathway")
def save_compressed_model(
    model: torch.nn.Module,
    filename: str,
    compression_format: CompressionFormat | None = None,
    force_contiguous: bool = True,
):
    raise NotImplementedError()
