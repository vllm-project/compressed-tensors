# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from compressed_tensors.config import (
    CompressionFormat,
    SparsityCompressionConfig,
    SparsityStructure,
)


__all__ = ["Sparse24BitMaskConfig"]


@SparsityCompressionConfig.register(name=CompressionFormat.sparse_24_bitmask.value)
class Sparse24BitMaskConfig(SparsityCompressionConfig):
    """
    Configuration for storing a 24 sparse model using
    bytemask compression

    :param global_sparsity: average sparsity of the entire model
    :param sparsity_structure: structure of the sparsity, should always be
        "2:4" for this compression format
    """

    format: str = CompressionFormat.sparse_24_bitmask.value
    global_sparsity: Optional[float] = 0.0
    sparsity_structure: Optional[str] = SparsityStructure.TWO_FOUR.value
