# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class PackedQuantizationCompressor(FormatCompressor):
    format = CompressionFormat.pack_quantized.value


# Backward-compatible alias
PackFormatCompressor = PackedQuantizationCompressor

__all__ = ["PackedQuantizationCompressor", "PackFormatCompressor"]
