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

import torch
from compressed_tensors.quantization import round_to_quantized_type_dtype
from compressed_tensors.quantization.utils import (
    generate_mxfp4_scales,
    maybe_convert_from_mxfp4_exp,
    round_to_power_2,
)


def test_round_power_2_noise():
    powers = torch.Tensor(
        [
            [2**-10, 2**-9, 2**-8, 2**-7, 2**-6],
            [2**-5, 2**-4, 2**-3, 2**-2, 2**-1],
            [2**0, 2**1, 2**-10, 2**-9, 2**-8],
            [2**-7, 2**-6, 2**-5, 2**-4, 2**-3],
            [2**-2, 2**-1, 2**0, 2**1, 2**-10],
        ]
    ).to(torch.bfloat16)

    noise = torch.rand_like(powers) * 0.2
    powers_noisy = powers * (1 + noise)
    rounded = round_to_power_2(powers_noisy)
    assert torch.equal(rounded, powers)


def test_round_power_2():
    x = torch.Tensor(
        (
            [5.687891, -8.291567, -1.540329, -0.315635, 0.965272],
            [-6.944130, 0.073246, -0.451778, 8.571118, -9.856593],
            [-0.040571, -0.708509, 2.485657, -4.003352, -0.995600],
            [0.224199, 5.032586, -1.309816, -0.621958, 7.290238],
            [-9.848001, -0.290731, 1.501562, 0.379829, -5.312081],
        )
    ).to(torch.bfloat16)
    x_rounded = torch.Tensor(
        (
            [4.000000, -8.000000, -1.000000, -0.250000, 1.000000],
            [-4.000000, 0.062500, -0.500000, 8.000000, -8.000000],
            [-0.0312, -0.500000, 2.000000, -4.000000, -1.000000],
            [0.250000, 4.000000, -1.000000, -0.500000, 8.000000],
            [-8.000000, -0.250000, 1.000000, 0.250000, -4.000000],
        )
    ).to(torch.bfloat16)
    rounded = round_to_power_2(x)
    assert torch.equal(rounded, x_rounded)


def test_mxfp4_scales_e2e():
    from compressed_tensors.quantization.quant_args import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    mock_weight = torch.normal(mean=0.0002, std=0.0576, size=(2880, 2880))

    x = mock_weight.reshape(*mock_weight.shape[:-1], -1, 32).to(torch.bfloat16)
    min_vals = torch.amin(x, dim=-1)
    max_vals = torch.amax(x, dim=-1)

    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
    block_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))

    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )

    scales = generate_mxfp4_scales(block_max)
    scales = round_to_quantized_type_dtype(scales, dtype=args.scale_dtype)

    converted_ct = maybe_convert_from_mxfp4_exp(args=args, scale=scales)

    scales_exp = torch.log2(converted_ct)
    block_max_exp = torch.floor(torch.log2(round_to_power_2(block_max)))
    assert torch.equal(scales_exp, block_max_exp)
