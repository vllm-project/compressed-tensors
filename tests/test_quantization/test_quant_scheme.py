# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
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
