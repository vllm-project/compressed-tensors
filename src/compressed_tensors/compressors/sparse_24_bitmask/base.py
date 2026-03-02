# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class Sparse24BitMaskCompressor(FormatCompressor):
    format = CompressionFormat.sparse_24_bitmask.value


# Backward-compatible alias
Sparse24BitmaskFormatCompressor = Sparse24BitMaskCompressor

__all__ = ["Sparse24BitMaskCompressor", "Sparse24BitmaskFormatCompressor"]
