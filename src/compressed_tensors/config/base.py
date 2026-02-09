# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, unique
from typing import List, Optional

from compressed_tensors.registry import RegistryMixin
from pydantic import BaseModel


__all__ = ["SparsityCompressionConfig", "CompressionFormat", "SparsityStructure"]


@unique
class CompressionFormat(Enum):
    dense = "dense"
    sparse_bitmask = "sparse-bitmask"
    sparse_24_bitmask = "sparse-24-bitmask"
    int_quantized = "int-quantized"
    float_quantized = "float-quantized"
    naive_quantized = "naive-quantized"
    pack_quantized = "pack-quantized"
    marlin_24 = "marlin-24"
    mixed_precision = "mixed-precision"
    nvfp4_pack_quantized = "nvfp4-pack-quantized"
    mxfp4_pack_quantized = "mxfp4-pack-quantized"


@unique
class SparsityStructure(Enum):
    """
    An enumeration to represent different sparsity structures.

    Attributes
    ----------
    TWO_FOUR : str
        Represents a 2:4 sparsity structure.
    ZERO_ZERO : str
        Represents a 0:0 sparsity structure.
    UNSTRUCTURED : str
        Represents an unstructured sparsity structure.

    Examples
    --------
    >>> SparsityStructure('2:4')
    <SparsityStructure.TWO_FOUR: '2:4'>

    >>> SparsityStructure('unstructured')
    <SparsityStructure.UNSTRUCTURED: 'unstructured'>

    >>> SparsityStructure('2:4') == SparsityStructure.TWO_FOUR
    True

    >>> SparsityStructure('UNSTRUCTURED') == SparsityStructure.UNSTRUCTURED
    True

    >>> SparsityStructure(None) == SparsityStructure.UNSTRUCTURED
    True

    >>> SparsityStructure('invalid')
    Traceback (most recent call last):
        ...
    ValueError: invalid is not a valid SparsityStructure
    """

    TWO_FOUR = "2:4"
    UNSTRUCTURED = "unstructured"
    ZERO_ZERO = "0:0"

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value.lower() if value is not None else value
        return obj

    @classmethod
    def _missing_(cls, value):
        # Handle None and case-insensitive values
        if value is None:
            return cls.UNSTRUCTURED
        for member in cls:
            if member.value == value.lower():
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class SparsityCompressionConfig(RegistryMixin, BaseModel):
    """
    Base data class for storing sparsity compression parameters

    :param format: name of compression format
    :param targets: List of layer names or layer types that aren't sparse and should
        be ignored during compression. By default, assume all layers are targeted
    :param ignore: List of layer names (unique) to ignore from targets. Defaults to None
    :param global_sparsity: average sparsity of the entire model
    :param sparsity_structure: structure of the sparsity, such as
    "unstructured", "2:4", "8:16" etc
    """

    format: str
    targets: Optional[List[str]] = None
    ignore: Optional[List[str]] = None
    global_sparsity: Optional[float] = 0.0
    sparsity_structure: Optional[str] = "unstructured"
