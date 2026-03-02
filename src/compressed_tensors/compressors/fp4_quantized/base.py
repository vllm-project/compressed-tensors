# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.fp4_quantized.impl import (
    MXFP4PackedCompressor,
    NVFP4PackedCompressor,
)
from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class NVFP4FormatCompressor(FormatCompressor):
    format = CompressionFormat.nvfp4_pack_quantized.value


class MXFP4FormatCompressor(FormatCompressor):
    format = CompressionFormat.mxfp4_pack_quantized.value


__all__ = [
    "NVFP4PackedCompressor",
    "MXFP4PackedCompressor",
    "NVFP4FormatCompressor",
    "MXFP4FormatCompressor",
]
