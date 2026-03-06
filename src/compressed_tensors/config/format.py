# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import deprecated
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
    if len(formats) >= 2:
        return CompressionFormat.mixed_precision
    if len(formats) == 1:
        return formats[0]
    else:
        return CompressionFormat.dense


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
    """
    formats = set()

    for name, module in model.named_modules(remove_duplicate=True):
        if not is_module_quantized(module):
            continue

        # infer format using priority list
        scheme: QuantizationScheme = module.quantization_scheme
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
        formats.add(format)

    return list(formats)


def get_module_format(
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
            if BaseCompressor.get_value_from_registry(format.value).match(
                module_type, scheme
            )
        )
    )


@deprecated("get_module_format")
def _get_quant_compression_format(
    input_args: Optional[QuantizationArgs],
    weight_args: Optional[QuantizationArgs],
    sparsity_structure: Optional[str] = None,
) -> CompressionFormat:
    """
    Using the weight and input quantization args as well as an optional
    sparsity structure, determine the compression format that should be
    applied to a given module

    :param input_args: input quantization parameters
    :param weight_args: weight quantization parameters
    :param sparsity_structure: optional (global) modle sparsity
        structure
    :return CompresssionFormat for the module
    """
    is_weight_only = weight_args is not None and input_args is None

    if weight_args.num_bits == 4 and weight_args.type == QuantizationType.FLOAT.value:
        if weight_args.group_size == 32:
            return CompressionFormat.mxfp4_pack_quantized
        return CompressionFormat.nvfp4_pack_quantized

    if is_weight_only:  # w4a16 and w8a16
        is_valid_pack = (
            weight_args.num_bits in [4, 8]
            and weight_args.type == QuantizationType.INT.value
        )
        if not is_valid_pack:  # packing only valid for int4 and int 8
            return CompressionFormat.naive_quantized

        return CompressionFormat.pack_quantized

    else:  # w8a8 float and int
        if (
            weight_args.type == QuantizationType.FLOAT.value
            and weight_args.num_bits == 8
        ):
            return CompressionFormat.float_quantized
        if weight_args.type == QuantizationType.INT.value:
            return CompressionFormat.int_quantized

        return CompressionFormat.naive_quantized
