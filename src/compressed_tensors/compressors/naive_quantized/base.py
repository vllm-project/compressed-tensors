# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.quantized_compressors.naive_quantized import (
    FloatQuantizationCompressor,
    IntQuantizationCompressor,
    NaiveQuantizationCompressor,
)
from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class NaiveFormatCompressor(FormatCompressor):
    format = CompressionFormat.naive_quantized.value


class IntFormatCompressor(FormatCompressor):
    format = CompressionFormat.int_quantized.value


class FloatFormatCompressor(FormatCompressor):
    format = CompressionFormat.float_quantized.value

__all__ = [
    "NaiveQuantizationCompressor",
    "IntQuantizationCompressor",
    "FloatQuantizationCompressor",
    "NaiveFormatCompressor",
    "IntFormatCompressor",
    "FloatFormatCompressor",
]
