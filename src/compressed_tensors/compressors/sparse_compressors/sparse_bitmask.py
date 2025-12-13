# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple, Union

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    merge_names,
    pack_bitmasks,
    register_offload_parameter,
    unpack_bitmasks,
)
from torch import Tensor


__all__ = [
    "BitmaskCompressor",
    "BitmaskTensor",
    "bitmask_compress",
    "bitmask_decompress",
]


@BaseCompressor.register(name=CompressionFormat.sparse_bitmask.value)
class BitmaskCompressor(BaseCompressor):
    """
    Compression for sparse models using bitmasks. Non-zero weights are stored in a 1d
    values tensor, with their locations stored in a 2d bitmask
    """

    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        # any module can be sparse
        return True

    @classmethod
    def compress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        assert hasattr(module, "weight")
        assert hasattr(module, "weight_scale")

        with align_module_device(module):
            bitmask_tensor = BitmaskTensor.from_dense(module.weight).dict("")
            for key, value in bitmask_tensor.items():
                value = torch.nn.Parameter(value, requires_grad=False)
                register_offload_parameter(module, key, value)

            delete_offload_parameter(module, "weight")

    @classmethod
    def decompress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        assert hasattr(module, "shape")
        assert hasattr(module, "compressed")
        assert hasattr(module, "bitmask")

        with align_module_device(module):
            weight = BitmaskTensor.from_compressed_data(
                shape=module.shape,
                compressed=module.compressed,
                bitmask=module.bitmask,
            ).decompress()

            delete_offload_parameter(module, "shape")
            delete_offload_parameter(module, "compressed")
            delete_offload_parameter(module, "bitmask")
            register_offload_parameter(module, "weight", weight)

    @property
    def format(self) -> CompressionFormat:
        return CompressionFormat.sparse_bitmask


class BitmaskTensor:
    """
    Owns compressions and decompression for a single bitmask compressed tensor.
    Adapted from: https://github.com/mgoin/torch_bitmask/tree/main

    :param shape: shape of dense tensor
    :compressed: flat tensor of non-zero values
    :bitmask: 2d bitmask of non-zero values
    :row_offsets: flat tensor indicating what index in values each dense row starts at
    """

    def __init__(
        self,
        shape: Union[torch.Size, List],
        compressed: Tensor,
        bitmask: Tensor,
        row_offsets: Tensor,
    ):
        self.shape = list(shape)
        self.compressed = compressed
        self.bitmask = bitmask
        self.row_offsets = row_offsets

    @staticmethod
    def from_dense(tensor: Tensor) -> "BitmaskTensor":
        """
        :param tensor: dense tensor to compress
        :return: instantiated compressed tensor
        """
        shape = tensor.shape
        compressed, bitmask, row_offsets = bitmask_compress(tensor.cpu())
        return BitmaskTensor(
            shape=shape, compressed=compressed, bitmask=bitmask, row_offsets=row_offsets
        )

    def decompress(self) -> Tensor:
        """
        :return: reconstructed dense tensor
        """
        return bitmask_decompress(self.compressed, self.bitmask, self.shape)

    def curr_memory_size_bytes(self):
        """
        :return: size in bytes required to store compressed tensor on disk
        """

        def sizeof_tensor(a):
            return a.element_size() * a.nelement()

        return (
            sizeof_tensor(self.compressed)
            + sizeof_tensor(self.bitmask)
            + sizeof_tensor(self.row_offsets)
        )

    def dict(self, name_prefix: str, device: str = "cpu") -> Dict[str, Tensor]:
        """
        :name_prefix: name of original tensor to store compressed weight as
        :return: dict of compressed data for the stored weight
        """
        return {
            merge_names(name_prefix, "shape"): torch.tensor(self.shape, device=device),
            merge_names(name_prefix, "compressed"): self.compressed.to(device),
            merge_names(name_prefix, "bitmask"): self.bitmask.to(device),
            merge_names(name_prefix, "row_offsets"): self.row_offsets.to(device),
        }

    def __repr__(self):
        return f"BitmaskTensor(shape={self.shape}, compressed=True)"


def bitmask_compress(tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compresses a dense tensor using bitmask compression

    :param tensor: dense tensor to compress
    :return: tuple of compressed data representing tensor
    """
    bytemasks = tensor != 0
    row_counts = bytemasks.sum(dim=-1)
    row_offsets = torch.cumsum(row_counts, 0) - row_counts
    if tensor.dtype == torch.float8_e4m3fn:
        # acces raw bytes of the tensor
        tensor_view = tensor.view(torch.int8)
        values = tensor_view[bytemasks]
        values = values.view(torch.float8_e4m3fn)
    else:
        values = tensor[bytemasks]
    bitmasks_packed = pack_bitmasks(bytemasks)
    return values, bitmasks_packed, row_offsets


def bitmask_decompress(
    values: Tensor, bitmasks: Tensor, original_shape: torch.Size
) -> Tensor:
    """
    Reconstructs a dense tensor from a compressed one

    :param values: 1d tensor of non-zero values
    :param bitmasks: 2d int8 tensor flagging locations of non-zero values in the
    tensors original shape
    :param original_shape: shape of the dense tensor
    :return: decompressed dense tensor
    """
    bytemasks_unpacked = unpack_bitmasks(bitmasks, original_shape)

    decompressed_tensor = torch.zeros(original_shape, dtype=values.dtype)
    decompressed_tensor[bytemasks_unpacked] = values

    return decompressed_tensor
