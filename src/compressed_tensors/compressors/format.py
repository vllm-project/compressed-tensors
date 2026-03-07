# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger


__all__ = [
    "flatten_formats",
    "infer_set_module_formats",
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
    if len(formats) <= 0:
        return CompressionFormat.dense
    if len(formats) == 1:
        return formats[0]
    if len(formats) >= 2:
        return CompressionFormat.mixed_precision


def infer_set_module_formats(
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
    :return: list of formats applied to modules (excluding dense format)
    """
    formats = set()

    for _, module in model.named_modules(remove_duplicate=True):
        if not is_module_quantized(module):
            continue

        # infer format using priority list
        scheme: "QuantizationScheme" = module.quantization_scheme
        format = get_module_format(type(module), scheme)

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
        if format != CompressionFormat.dense:
            formats.add(format)

    return list(formats)


def get_module_format(
    module_type: type, scheme: "QuantizationScheme"
) -> CompressionFormat:
    """
    Infer the module's compression format using the module's type and quant scheme

    :param module_type: module type, typically linear
    :param scheme: quantization applied to module
    :return: format that should be used to compress the module
    """
    # avoid circular imports
    from compressed_tensors.compressors import BaseCompressor

    return next(
        (
            format
            for format in COMPRESSION_FORMAT_PRIORITY
            if BaseCompressor.get_value_from_registry(format.value).match(
                module_type, scheme
            )
        )
    )
