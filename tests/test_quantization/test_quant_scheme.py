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
