# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from typing import Dict, Tuple

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from compressed_tensors.utils.offload import get_execution_device
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import linear
from torch.nn.modules import Linear


class CompressedLinear(Linear):
    """
    Wrapper module for running a compressed forward pass of a quantized Linear module.
    The wrapped layer will decompressed on each forward call.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "CompressedLinear should not be initialized directly. "
            "Use the from_linear method instead.",
            UserWarning,
        )

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        module: Linear,
        quantization_scheme: QuantizationScheme,
        quantization_format: str,
    ):
        """
        :param module: dense linear module to replace
        :param quantization_scheme: quantization config for the module to wrap
        :param quantization_format: compression format module is stored as
        :return: CompressedLinear module wrapping the input module
        """
        module.__class__ = CompressedLinear
        module.compressor = BaseCompressor.load_from_registry(quantization_format)
        init_device = get_execution_device(module)

        # this will initialize all the scales and zero points
        initialize_module_for_quantization(
            module, quantization_scheme, force_zero_point=False
        )

        # get the shape and dtype of compressed parameters
        compression_params: Dict[str, Tuple] = module.compressor.compression_param_info(
            module.weight.shape, quantization_scheme.weights
        )

        # no need for this once quantization is initialized, will be replaced
        # with the compressed parameter
        delattr(module, "weight")

        # populate compressed weights and quantization parameters
        for name, (shape, dtype) in compression_params.items():
            param = Parameter(
                torch.empty(shape, device=init_device, dtype=dtype), requires_grad=False
            )
            module.register_parameter(name, param)

        # mark module as compressed
        module.quantization_status = QuantizationStatus.COMPRESSED

        return module

    def forward(self, input: Tensor) -> Tensor:
        """
        Decompresses the weight, then runs the quantized forward pass
        """
        # Import here to avoid circular dependency
        from compressed_tensors.quantization.lifecycle.forward import forward_quantize

        if self.quantization_status == QuantizationStatus.COMPRESSED:
            weight_data = self.compressor.decompress_module(self)
            param = Parameter(weight_data, requires_grad=False)
            self.register_parameter("weight", param)

            self.quantization_status = QuantizationStatus.FROZEN

        # Apply quantization if enabled
        scheme = getattr(self, "quantization_scheme", None)
        status = getattr(self, "quantization_status", None)
        enabled = (
            getattr(self, "quantization_enabled", True)
            and scheme is not None
            and status is not None
        )
        weight = self.weight
        bias = self.bias

        if enabled:
            if scheme.input_activations is not None:
                input = forward_quantize(self, input, "input", scheme.input_activations)

            # Don't quantize weight for FROZEN status (already quantized during compression)
            if scheme.weights is not None and status is not QuantizationStatus.FROZEN:
                weight = forward_quantize(self, weight, "weight", scheme.weights)

        output = linear(input, weight, bias)

        if enabled and scheme.output_activations is not None:
            output = forward_quantize(self, output, "output", scheme.output_activations)

        return output
