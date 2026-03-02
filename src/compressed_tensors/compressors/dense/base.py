# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.sparse_compressors.dense import DenseCompressor
from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class DenseFormatCompressor(FormatCompressor):
    format = CompressionFormat.dense.value


__all__ = ["DenseCompressor", "DenseFormatCompressor"]
