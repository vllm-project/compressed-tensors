# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class BitmaskCompressor(FormatCompressor):
    format = CompressionFormat.sparse_bitmask.value


# Backward-compatible alias
SparseBitmaskFormatCompressor = BitmaskCompressor

__all__ = ["BitmaskCompressor", "SparseBitmaskFormatCompressor"]
