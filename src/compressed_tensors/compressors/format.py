# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger


__all__ = ["infer_model_format", "infer_module_format"]


# Priority order for compression format matching
# More specific formats should come before more general ones
COMPRESSION_FORMAT_PRIORITY: list[CompressionFormat] = [
    CompressionFormat.mxfp4_pack_quantized,
    CompressionFormat.mxfp8_quantized,
    CompressionFormat.nvfp4_pack_quantized,
    CompressionFormat.pack_quantized,
    CompressionFormat.int_quantized,
    CompressionFormat.float_quantized,
    CompressionFormat.naive_quantized,
    CompressionFormat.dense,
]


def infer_model_format(
    model: torch.nn.Module,
    force_compression_format: str | None = None,
) -> CompressionFormat:
    """
    Infers the quantization format for a model based on its modules

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
        scheme: QuantizationScheme = module.quantization_scheme
        format = infer_module_format(type(module), scheme)

        # user provides a global override format
        if force_compression_format is not None:
            if force_compression_format != format.value:
                logger.warning(
                    f"The provided format {force_compression_format} does not match "
                    f"the inferred format {format.value}. Compression may fail",
                    log_once=True,
                )
            format = force_compression_format

        # user provides a format via QuantizationScheme.format
        elif scheme.format is not None:
            format = scheme.format

        scheme.format = CompressionFormat(format)
        if format != CompressionFormat.dense:
            formats.add(format)

    return _flatten_formats(formats)


def infer_module_format(
    module_type: type, scheme: QuantizationScheme
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
            if BaseCompressor.get_value_from_registry(format.value).can_compress(
                module_type, scheme
            )
        )
    )


def _flatten_formats(formats: set[CompressionFormat]) -> CompressionFormat:
    """
    Reduce a list of compression formats to a single summary format.

    Returns dense if the list is empty, the single format if there's only one,
    or mixed_precision if there are multiple different formats.

    :param formats: list of compression formats found in the model
    :return: single compression format representing the overall model
    """
    if len(formats) <= 0:
        return CompressionFormat.dense
    if len(formats) == 1:
        return list(formats)[0]
    if len(formats) >= 2:
        return CompressionFormat.mixed_precision
