# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Protocol

import torch
from compressed_tensors.quantization import QuantizationConfig


class Converter(Protocol):
    """
    Converter interface, to modify safetensors files based on tensor name and
    pointer to torch.Tensor, and create the QuantizationConfig
    """

    def process(self, tensors: dict[str, torch.Tensor]):
        """
        Operate on safetensors file in-place, to convert it into a compressed-tensors
        compatible format.
        e.g. rename tensor, or invert weights to match compressed-tensors convention.

        :param tensors: dictionary of tensor name to tensor, as loaded from
        safetensors file. Tensor name is a concatenation of module name and
        parameter name, e.g.
        - `model.layers.0.self_attn.q_proj.weight`
        - `model.layers.0.mlp.up_proj.weight_packed`
        """
        pass

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Validation layer to quickly log warnings or raise an error if the safetensors
        file is not compatible with Converter.

        :param tensors: dictionary of tensor name to tensor, as loaded from
        safetensors file.
        """
        pass

    def create_config(self) -> QuantizationConfig:
        """
        Create compressed-tensors QuantizationConfig so that it can be set in the
        new model checkpoint's config.json.
        """
        pass
