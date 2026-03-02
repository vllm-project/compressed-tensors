# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.sparse_compressors.sparse_24_bitmask import (
    Sparse24BitMaskCompressor,
    Sparse24BitMaskTensor,
)
from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class Sparse24BitmaskFormatCompressor(FormatCompressor):
    format = CompressionFormat.sparse_24_bitmask.value


__all__ = [
    "Sparse24BitMaskCompressor",
    "Sparse24BitMaskTensor",
    "Sparse24BitmaskFormatCompressor",
]
