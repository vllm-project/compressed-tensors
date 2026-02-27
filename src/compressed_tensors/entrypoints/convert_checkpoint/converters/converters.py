# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Protocol

import torch

from compressed_tensors.quantization import (
    QuantizationConfig,
)


class Converter(Protocol):
    """
    Converter interface, to modify safetensors files based
    on tensor name and pointer to torch.Tensor
    """

    def process(self, tensors: dict[str, torch.Tensor]):
        pass

    def validate(self, tensors: dict[str, torch.Tensor]):
        pass

    def create_config(self) -> QuantizationConfig:
        pass
