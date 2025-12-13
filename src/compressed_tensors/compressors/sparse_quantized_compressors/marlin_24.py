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

import logging

import numpy as np
import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.compressed import quantize_weight
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    get_permutations_24,
    getattr_chain,
    register_offload_parameter,
    sparse_semi_structured_from_dense_cutlass,
)
from torch import Tensor


_LOGGER: logging.Logger = logging.getLogger(__name__)


@BaseCompressor.register(name=CompressionFormat.marlin_24.value)
class Marlin24Compressor(BaseCompressor):
    """
    Compresses a quantized model with 2:4 sparsity structure for inference with the
    Marlin24 kernel. Decompression is not implemented for this compressor.
    """

    @staticmethod
    def match_scheme(scheme: QuantizationScheme) -> bool:
        return (
            scheme.weights is not None
            and scheme.weights.strategy
            in (QuantizationStrategy.GROUP, QuantizationStrategy.CHANNEL)
            and scheme.weights.group_size == 128
            and scheme.weights.symmetric
            and scheme.format == CompressionFormat.marlin_24
        )

    @classmethod
    def compress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        assert hasattr(module, "weight")
        assert hasattr(module, "weight_scale")
        args: QuantizationArgs = getattr_chain(module, "quantization_scheme.weights")

        with align_module_device(module):
            # Marlin24 kernel requires float16 inputs
            value = module.weight.to(torch.float16)
            scale = module.weight_scale.to(torch.float16)

            # quantize weight, keeping it as a float16 for now
            value = quantize_weight(module, value)

            # compress based on sparsity structure
            value, meta = compress_weight_24(value)
            meta = meta.cpu()

            # TODO: why does this have to be on CPU?
            # Marlin24 kernel expects input dim first
            value = value.t().contiguous().cpu()
            scale = scale.t().contiguous().cpu()
            og_weight_shape = value.shape

            # Marlin24 kernel expects unsigned values, shift zero-point
            value += (1 << args.num_bits) // 2

            # pack quantized weight and scale
            value = pack_weight_24(value, args)
            packed_scale = pack_scales_24(scale, args, og_weight_shape)
            meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

            # save compressed values
            delete_offload_parameter(module, "weight")
            register_offload_parameter(module, "weight_packed", value)
            register_offload_parameter(module, "scale_packed", packed_scale)
            register_offload_parameter(module, "meta", meta)

    @classmethod
    def decompress_module(cls, module: torch.nn.Linear) -> torch.nn.Linear:
        raise NotImplementedError(
            "Decompression is not implemented for the Marlin24 Compressor."
        )

    @property
    def format(self) -> CompressionFormat:
        return CompressionFormat.marlin_24


def compress_weight_24(weight: Tensor):
    weight = weight.contiguous()
    w_comp, meta = sparse_semi_structured_from_dense_cutlass(weight)
    w_comp = w_comp.contiguous()
    return w_comp, meta


def marlin_permute_weights(q_w, size_k, size_n, perm, tile):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def pack_weight_24(
    weight: Tensor,
    quantization_args: QuantizationArgs,
    tile: int = 16,
) -> Tensor:
    size_k = weight.shape[0]
    size_n = weight.shape[1]
    num_bits = quantization_args.num_bits
    pack_factor = 32 // num_bits

    # Reshuffle to marlin_24 format
    perm, _, _ = get_permutations_24(num_bits)
    q_w = marlin_permute_weights(weight, size_k, size_n, perm, tile)

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32))

    return q_packed


def pack_scales_24(
    scales: Tensor, quantization_args: QuantizationArgs, w_shape: torch.Size
) -> Tensor:
    size_k = w_shape[0]
    size_n = w_shape[1]
    num_bits = quantization_args.num_bits

    _, scale_perm_2_4, scale_perm_single_2_4 = get_permutations_24(num_bits)

    if (
        quantization_args.strategy == QuantizationStrategy.GROUP
        and quantization_args.group_size < size_k
    ):
        scales = scales.reshape((-1, len(scale_perm_2_4)))[:, scale_perm_2_4]
    else:  # channelwise
        scales = scales.reshape((-1, len(scale_perm_single_2_4)))[
            :, scale_perm_single_2_4
        ]
    scales = scales.reshape((-1, size_n)).contiguous()

    return scales
