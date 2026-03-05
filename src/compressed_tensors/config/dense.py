# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig


__all__ = ["DenseSparsityConfig"]


@SparsityCompressionConfig.register(name=CompressionFormat.dense.value)
class DenseSparsityConfig(SparsityCompressionConfig):
    """
    Identity configuration for storing a sparse model in
    an uncompressed dense format

    :param global_sparsity: average sparsity of the entire model
    :param sparsity_structure: structure of the sparsity, such as
    "unstructured", "2:4", "8:16" etc
    """

    format: str = CompressionFormat.dense.value
    global_sparsity: float | None = 0.0
    sparsity_structure: str | None = "unstructured"
