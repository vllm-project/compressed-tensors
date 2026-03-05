# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict

import pytest
import torch
from compressed_tensors.config.format import infer_set_module_format
from compressed_tensors.quantization import preset_name_to_scheme


@pytest.mark.parametrize(
    "preset,expected_format",
    [
        ["W8A8", "int-quantized"],
        ["W8A16", "pack-quantized"],
        ["W4A16", "pack-quantized"],
        ["FP8", "float-quantized"],
    ],
)
def test_infer_quant_format(preset, expected_format):
    quant_scheme = preset_name_to_scheme(preset, targets=["Linear"])

    dummy_model = torch.nn.Sequential(
        OrderedDict(
            [
                ("fc1", torch.nn.Linear(8, 16, bias=True)),
                ("fc2", torch.nn.Linear(16, 32, bias=True)),
                (
                    "block1",
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                ("fc1", torch.nn.Linear(32, 16, bias=True)),
                                ("fc2", torch.nn.Linear(16, 8, bias=True)),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    for _, module in dummy_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.quantization_scheme = quant_scheme

    inferred_formats = infer_set_module_format(dummy_model)
    assert inferred_formats[0].value == expected_format
