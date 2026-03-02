# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.sparse_bitmask.impl import (
    BitmaskCompressor,
    BitmaskTensor,
)
from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class SparseBitmaskFormatCompressor(FormatCompressor):
    format = CompressionFormat.sparse_bitmask.value


__all__ = ["BitmaskCompressor", "BitmaskTensor", "SparseBitmaskFormatCompressor"]
