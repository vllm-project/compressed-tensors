# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.mxfp6 import MXFP6PackedCompressor
from compressed_tensors.compressors.mxfp6.helpers import (
    kE2M3ToFloat,
    pack_fp6_to_uint8,
    unpack_fp6_from_uint8,
)
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams


def test_pack_fp6_layout():
    # codes 0b001000, 0b100001, 0b011111, 0b000000 packed little-endian
    values = torch.tensor([[1.0, -0.125, 7.5, 0.0]])
    assert pack_fp6_to_uint8(values).tolist() == [[0x48, 0xF8, 0x01]]


def test_pack_unpack_fp6_all_codes():
    values = torch.cat([kE2M3ToFloat, -kE2M3ToFloat]).reshape(4, 16)
    unpacked = unpack_fp6_from_uint8(pack_fp6_to_uint8(values), 4, 16, torch.float32)
    assert torch.equal(unpacked, values)
    assert torch.equal(torch.signbit(unpacked), torch.signbit(values))


def test_mxfp6_compress_decompress():
    scheme = preset_name_to_scheme("MXFP6A8", ["Linear"])
    weight = torch.randn(128, 256, dtype=torch.bfloat16) * 0.05
    groups = weight.reshape(128, 8, 32)
    scale, _ = calculate_qparams(groups.amin(-1), groups.amax(-1), scheme.weights)

    compressed = MXFP6PackedCompressor.compress(
        {"weight": weight, "weight_scale": scale}, scheme
    )
    assert compressed["weight_packed"].shape == (128, 192)
    assert compressed["weight_scale"].dtype == torch.uint8

    decompressed = MXFP6PackedCompressor.decompress(compressed, scheme)
    assert torch.equal(decompressed["weight_scale"], scale)
    expected = fake_quantize(weight, scale, None, scheme.weights)
    assert torch.equal(decompressed["weight"], expected)
