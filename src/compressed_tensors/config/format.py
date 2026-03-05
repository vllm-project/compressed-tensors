# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger


__all__ = [
    "flatten_formats",
    "infer_set_module_format",
]


# Priority order for compression format matching
# More specific formats should come before more general ones
COMPRESSION_FORMAT_PRIORITY: List[CompressionFormat] = [
    CompressionFormat.mxfp4_pack_quantized,
    CompressionFormat.nvfp4_pack_quantized,
    CompressionFormat.pack_quantized,
    CompressionFormat.int_quantized,
    CompressionFormat.float_quantized,
    CompressionFormat.naive_quantized,
    CompressionFormat.dense,
]


def flatten_formats(formats: list[CompressionFormat]) -> CompressionFormat:
    if len(formats) >= 2:
        return CompressionFormat.mixed_precision
    if len(formats) == 1:
        return formats[0]
    else:
        return CompressionFormat.dense


def infer_set_module_format(
    model: torch.nn.Module,
    force_compression_format: Optional[str] = None,
) -> None:
    """
    Infers the quantization format for a model based on its state and provided
    compression arguments. Updates the quantization_scheme.format value
    based on the inferred format.

    For a summary of the formats, see `docs/guides/compression_formats.md`.

    :param model: model to check for quantization
    :param quantization_format: optional global format to override
        the per module formats
    """
    # avoid circular imports
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationScheme

    formats = set()

    for name, module in model.named_modules(remove_duplicate=True):
        if not is_module_quantized(module):
            continue

        # infer format using priority list
        scheme: QuantizationScheme = module.quantization_scheme
        format = next(
            (
                fmt
                for fmt in COMPRESSION_FORMAT_PRIORITY
                if BaseCompressor.get_value_from_registry(fmt.value).match(module)
            )
        )

        # user provides a global override format
        if force_compression_format is not None:
            if force_compression_format != format.value:
                logger.warning(
                    "The provided format for the module does not match the "
                    "inferred format. Compression may fail "
                )
            format = force_compression_format

        # user provides a format via QuantizationScheme.format
        elif scheme.format is not None and scheme.format != format:
            logger.warning(
                "The provided format for the module does not match the "
                "inferred format. Compression may fail "
            )
            format = scheme.format

        scheme.format = CompressionFormat(format)
        formats.add(format)

    return list(formats)
