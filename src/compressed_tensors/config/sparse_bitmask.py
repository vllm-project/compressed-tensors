# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig


__all__ = ["BitmaskConfig"]


@SparsityCompressionConfig.register(name=CompressionFormat.sparse_bitmask.value)
class BitmaskConfig(SparsityCompressionConfig):
    """
    Configuration for storing a sparse model using
    bitmask compression

    :param global_sparsity: average sparsity of the entire model
    :param sparsity_structure: structure of the sparsity, such as
    "unstructured", "2:4", "8:16" etc
    """

    format: str = CompressionFormat.sparse_bitmask.value
    global_sparsity: Optional[float] = 0.0
    sparsity_structure: Optional[str] = "unstructured"
