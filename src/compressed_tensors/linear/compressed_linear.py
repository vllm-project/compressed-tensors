# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.quantization import QuantizationScheme


class CompressedLinear(torch.nn.Linear):
    """
    Wrapper module for running a compressed forward pass of a quantized Linear module.
    The wrapped layer will decompressed on each forward call.
    """

    @classmethod
    def from_linear(
        cls,
        module: torch.nn.Linear,
        quantization_scheme: QuantizationScheme,
        quantization_format: str,
    ):
        raise ValueError("`CompressedLinear` is no longer supported")
