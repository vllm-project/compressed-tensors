from copy import deepcopy

import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    apply_quantization_config,
    disable_quantization,
    enable_quantization,
)
from torch.nn import Linear


def test_quantization_enabled_disabled():
    inp = torch.randn(16)
    model = Linear(16, 16)
    quantized_model = deepcopy(model)
    apply_quantization_config(
        model=quantized_model,
        config=QuantizationConfig(
            config_groups=dict(W8A8=["Linear"]),
            quantization_status="calibration",
        ),
    )

    # run one calibration pass
    quantized_model(inp)

    model_output = model(inp)
    quantized_model_output = quantized_model(inp)

    # quantized and non quantized outputs should be different
    assert not torch.all(model_output == quantized_model_output)

    # disable quantization
    quantized_model.apply(disable_quantization)
    # check that quantized model now matches model output
    assert torch.all(model_output == quantized_model(inp))

    # re-enable quantization
    quantized_model.apply(enable_quantization)
    # check that quantized model matches original quantized output
    assert torch.all(quantized_model_output == quantized_model(inp))
