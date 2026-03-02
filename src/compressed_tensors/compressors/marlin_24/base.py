# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.compressors.marlin_24.impl import (
    Marlin24Compressor,
)
from compressed_tensors.compressors.format_compressor import FormatCompressor
from compressed_tensors.config import CompressionFormat


class Marlin24FormatCompressor(FormatCompressor):
    format = CompressionFormat.marlin_24.value


__all__ = ["Marlin24Compressor", "Marlin24FormatCompressor"]
