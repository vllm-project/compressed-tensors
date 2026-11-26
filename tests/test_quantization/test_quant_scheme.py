# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import ValidationError


def test_basic_scheme():
    targets = ["model.layer.0", "model.layer.3"]
    weights = QuantizationArgs()

    scheme = QuantizationScheme(targets=targets, weights=weights)
    assert scheme.targets == targets
    assert scheme.weights == weights
    assert scheme.input_activations is None
    assert scheme.output_activations is None
    assert scheme.format is None


def test_full_scheme():
    targets = ["Linear"]
    weights = QuantizationArgs()
    input_activations = QuantizationArgs(num_bits=8)
    output_activations = QuantizationArgs(num_bits=8, type="float", symmetric=False)

    scheme = QuantizationScheme(
        targets=targets,
        weights=weights,
        input_activations=input_activations,
        output_activations=output_activations,
        format="float-quantized",
    )
    assert scheme.targets == targets
    assert scheme.weights == weights
    assert scheme.input_activations == input_activations
    assert scheme.output_activations == output_activations
    assert scheme.format == "float-quantized"


def test_group_dynamic_input_activations_supported():
    # GROUP strategy dynamic input activations are handled by
    # compute_dynamic_scales_and_zp, so scheme validation must accept them
    # rather than rejecting with a "not supported" error (#758).
    input_activations = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
        dynamic=True,
    )
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, group_size=128),
        input_activations=input_activations,
    )
    assert scheme.input_activations.strategy == QuantizationStrategy.GROUP
    assert scheme.input_activations.dynamic is True


def test_unsupported_activation_strategy_still_rejected():
    # activation strategies outside the supported set remain rejected
    with pytest.raises(NotImplementedError):
        QuantizationScheme(
            targets=["Linear"],
            input_activations=QuantizationArgs(
                num_bits=8, strategy=QuantizationStrategy.CHANNEL
            ),
        )


def test_needs_targets():
    with pytest.raises(ValidationError):
        _ = QuantizationScheme()


def test_defaults():
    targets = ["Linear"]
    output = QuantizationScheme(targets=targets)
    assert output.weights is None
    assert output.input_activations is None
    assert output.output_activations is None
    assert output.format is None


def test_serialize_scheme():
    """Test serialization of QuantizationScheme including format"""
    from compressed_tensors.config import CompressionFormat

    targets = ["Linear"]
    weights = QuantizationArgs(num_bits=4, symmetric=True, group_size=128)
    input_activations = QuantizationArgs(num_bits=8, dynamic=True)
    output_activations = QuantizationArgs(num_bits=8, type="float", symmetric=False)

    scheme = QuantizationScheme(
        targets=targets,
        weights=weights,
        input_activations=input_activations,
        output_activations=output_activations,
        format=CompressionFormat.pack_quantized,
    )

    # Serialize to dict
    scheme_dict = scheme.model_dump()
    assert scheme_dict["targets"] == targets
    assert scheme_dict["format"] == "pack-quantized"
    assert "weights" in scheme_dict
    assert scheme_dict["weights"]["num_bits"] == 4
    assert "input_activations" in scheme_dict
    assert "output_activations" in scheme_dict

    # Deserialize from dict
    reloaded = QuantizationScheme.model_validate(scheme_dict)
    assert reloaded == scheme


def test_hif8_kv_preset():
    """HIF8_KV declares a static per-tensor FLOAT32 scale contract for HiF8 KV cache."""
    from compressed_tensors.quantization.quant_scheme import HIF8_KV

    assert HIF8_KV.num_bits == 8
    assert HIF8_KV.type == QuantizationType.FLOAT
    assert HIF8_KV.strategy == QuantizationStrategy.TENSOR
    assert HIF8_KV.symmetric is True
    assert HIF8_KV.dynamic is False
    assert HIF8_KV.observer == "static_minmax"
    assert HIF8_KV.scale_dtype == torch.float32


def test_hif8_kv_as_kv_cache_scheme():
    """HIF8_KV is accepted as a kv_cache_scheme and survives a serialize round-trip."""
    from compressed_tensors.quantization.quant_config import QuantizationConfig
    from compressed_tensors.quantization.quant_scheme import HIF8_KV

    config = QuantizationConfig(config_groups={}, kv_cache_scheme=HIF8_KV)
    assert config.kv_cache_scheme == HIF8_KV

    reloaded = QuantizationConfig.model_validate(config.model_dump())
    assert reloaded == config
    assert reloaded.kv_cache_scheme == HIF8_KV
