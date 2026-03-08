# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import FP8_E4M3_DATA
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.mark.parametrize(
    "frozen_stub,q_format,compressed_stub",
    [
        (
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-uncompressed",
            "float-quantized",
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-compressed",
        ),
        (
            "nm-testing/llama2.c-stories15M-ultrachat-mixed-uncompressed",
            "pack-quantized",
            "nm-testing/llama2.c-stories15M-ultrachat-mixed-compressed",
        ),
    ],
)
def test_compress_model(frozen_stub, q_format, compressed_stub):
    """Check that compression generates the expected compressed model"""
    model = AutoModelForCausalLM.from_pretrained(frozen_stub, torch_dtype=torch.float32)
    compressor = ModelCompressor.from_pretrained_model(model, None, q_format)
    true_compressed_model = AutoModelForCausalLM.from_pretrained(
        compressed_stub, torch_dtype=torch.float32
    )

    compressor.compress_model(model)
    compressed = dict(model.state_dict())
    true_compressed = dict(true_compressed_model.state_dict())

    assert compressed.keys() == true_compressed.keys()
    for key in compressed.keys():
        assert compressed[key].dtype == true_compressed[key].dtype, key
        assert torch.equal(compressed[key], true_compressed[key])


@pytest.mark.parametrize(
    "model_stub,q_format,compressed_stub",
    [
        (
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-uncompressed",
            "float-quantized",
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-compressed",
        ),
        (
            "nm-testing/llama2.c-stories15M-ultrachat-mixed-uncompressed",
            "pack-quantized",
            "nm-testing/llama2.c-stories15M-ultrachat-mixed-compressed",
        ),
    ],
)
def test_decompress_model(model_stub, q_format, compressed_stub):
    from transformers.utils.quantization_config import CompressedTensorsConfig

    model = AutoModelForCausalLM.from_pretrained(model_stub, torch_dtype=torch.float32)
    compressor = ModelCompressor.from_pretrained_model(model, None, q_format)
    true_decompressed_model = AutoModelForCausalLM.from_pretrained(
        compressed_stub,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
        torch_dtype=torch.float32,
    )

    compressor.compress_model(model)
    compressor.decompress_model(model)

    true_decompressed = dict(true_decompressed_model.state_dict())
    decompressed = dict(model.state_dict())

    assert decompressed.keys() == true_decompressed.keys()
    for key in decompressed.keys():
        assert decompressed[key].dtype == true_decompressed[key].dtype, key
        assert torch.equal(decompressed[key], true_decompressed[key])


def test_multiple_quant_compressors():
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Linear(2, 3))
    input_activations = QuantizationArgs(num_bits=8, type="float")
    weights = QuantizationArgs(num_bits=8, type="float")

    scheme_fp8 = QuantizationScheme(
        targets=["Linear"],
        weights=weights,
        input_activations=input_activations,
        format=CompressionFormat.float_quantized.value,
    )

    input_activations = QuantizationArgs(
        num_bits=4,
        type="float",
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    )
    weights = QuantizationArgs(
        num_bits=4,
        type="float",
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    )

    scheme_nvfp4 = QuantizationScheme(
        targets=["Linear"],
        weights=weights,
        input_activations=input_activations,
        format=CompressionFormat.nvfp4_pack_quantized.value,
    )

    model[0].quantization_scheme = scheme_fp8
    initialize_module_for_quantization(model[0])
    model[0].quantization_status = "frozen"
    model[1].quantization_scheme = scheme_nvfp4
    initialize_module_for_quantization(model[1])
    model[1].quantization_status = "frozen"

    compressor = ModelCompressor.from_pretrained_model(model, None)
    compressor.compress_model(model)
    assert compressor.quantization_config.format == CompressionFormat.mixed_precision
    assert model[0].quantization_scheme.format == scheme_fp8.format
    assert model[1].quantization_scheme.format == scheme_nvfp4.format


def test_compressed_model_inference_with_hook():
    model_stub = "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-compressed"

    # Load compressed model
    model = AutoModelForCausalLM.from_pretrained(
        model_stub, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_stub)

    # Model should have the decompression hook attached
    assert hasattr(model, "ct_decompress_hook")

    # Run a forward pass to trigger the hook
    prompt = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device=model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)

    # After forward pass, hook should have triggered and been removed
    assert not hasattr(model, "ct_decompress_hook")

    # Check perplexity is reasonable
    assert torch.exp(outputs.loss) <= 500.0
