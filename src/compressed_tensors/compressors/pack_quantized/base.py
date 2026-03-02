# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
    PackedQuantizationCompressor,
)
from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class PackFormatCompressor(FormatCompressor):
    format = CompressionFormat.pack_quantized.value


__all__ = ["PackedQuantizationCompressor", "PackFormatCompressor"]
