# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class Marlin24Compressor(FormatCompressor):
    format = CompressionFormat.marlin_24.value


# Backward-compatible alias
Marlin24FormatCompressor = Marlin24Compressor

__all__ = ["Marlin24Compressor", "Marlin24FormatCompressor"]
