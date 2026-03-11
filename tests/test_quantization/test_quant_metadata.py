# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.offload import offload_module
from compressed_tensors.quantization import (
    QuantizationMetadata,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)


@pytest.mark.parametrize("offloaded", (True, False))
def test_clear(offloaded):
    module = torch.nn.Linear(16, 16)
    scheme = preset_name_to_scheme("NVFP4", ["Linear"])
    base_forward = module.forward

    # offload module
    if offloaded:
        offload_module(module, "cpu", "cpu")
        offloaded_forward = module.forward
        assert module._original_forward_func is base_forward.__func__

    # add quantized forward (inside offloaded forward)
    initialize_module_for_quantization(module, scheme)
    qparams = ["weight_scale", "weight_global_scale", "input_global_scale"]

    for name in qparams:
        assert hasattr(module, name)

    if offloaded:
        assert module.forward is offloaded_forward
        assert module._original_forward_func.__wrapped__ is base_forward.__func__
    else:
        assert module.forward.__wrapped__ is base_forward.__func__

    # remove quantized forward
    QuantizationMetadata.clear_quantization(module)

    for name in qparams:
        assert not hasattr(module, name)

    if offloaded:
        assert module.forward is offloaded_forward
        assert module._original_forward_func is base_forward.__func__
    else:
        assert module.forward.__func__ is base_forward.__func__
