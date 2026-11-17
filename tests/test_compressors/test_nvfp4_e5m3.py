# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors import (
    NVFP4E5M3GSPackedCompressor,
    NVFP4E5M3PackedCompressor,
    NVFP4PackedCompressor,
)
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    GlobalScaleType,
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
    ScaleFormat,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils.ue5m3 import (
    ue5m3_decode,
    ue5m3_encode,
)


def test_preset_formats_and_aliases():
    no_gs = preset_name_to_scheme("NVFP4_E5M3", ["Linear"])
    assert no_gs == preset_name_to_scheme("NVFP4_UE5M3", ["Linear"])
    assert no_gs == preset_name_to_scheme("NVFP4+", ["Linear"])
    assert infer_module_format(torch.nn.Linear, no_gs) == (
        CompressionFormat.nvfp4_e5m3_pack_quantized
    )

    gs = preset_name_to_scheme("NVFP4_E5M3_GS", ["Linear"])
    assert gs == preset_name_to_scheme("NVFP4_UE5M3_GS", ["Linear"])
    assert gs == preset_name_to_scheme("NVFP4+_GS", ["Linear"])
    assert infer_module_format(torch.nn.Linear, gs) == (
        CompressionFormat.nvfp4_e5m3_gs_pack_quantized
    )

    standard = preset_name_to_scheme("NVFP4", ["Linear"])
    assert infer_module_format(torch.nn.Linear, standard) == (
        CompressionFormat.nvfp4_pack_quantized
    )


def test_compressor_selection_and_parameter_names():
    no_gs = preset_name_to_scheme("NVFP4_E5M3", ["Linear"])
    gs = preset_name_to_scheme("NVFP4_E5M3_GS", ["Linear"])

    assert NVFP4E5M3PackedCompressor.can_compress(torch.nn.Linear, no_gs)
    assert not NVFP4E5M3GSPackedCompressor.can_compress(torch.nn.Linear, no_gs)
    assert not NVFP4PackedCompressor.can_compress(torch.nn.Linear, no_gs)

    assert NVFP4E5M3GSPackedCompressor.can_compress(torch.nn.Linear, gs)
    assert not NVFP4E5M3PackedCompressor.can_compress(torch.nn.Linear, gs)

    assert NVFP4E5M3PackedCompressor.compression_param_names(no_gs) == (
        "weight_packed",
        "weight_scale",
    )
    assert NVFP4E5M3GSPackedCompressor.compression_param_names(gs) == (
        "weight_packed",
        "weight_scale",
        "weight_global_scale",
        "input_global_scale",
    )


def test_format_infers_e5m3_metadata_for_legacy_configs():
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=16,
        ),
        input_activations=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=16,
            dynamic="local",
        ),
        format=CompressionFormat.nvfp4_e5m3_pack_quantized,
    )

    assert scheme.weights.scale_format == ScaleFormat.UE5M3
    assert scheme.weights.global_scale == GlobalScaleType.NONE
    assert scheme.input_activations.scale_format == ScaleFormat.UE5M3
    assert scheme.input_activations.global_scale == GlobalScaleType.NONE
    assert infer_module_format(torch.nn.Linear, scheme) == (
        CompressionFormat.nvfp4_e5m3_pack_quantized
    )


def test_no_gs_compress_decompress_round_trip():
    scheme = preset_name_to_scheme("NVFP4_E5M3", ["Linear"])
    scale = torch.tensor(
        [[0.5, 1.0], [2.0, 8.0]],
        dtype=torch.float32,
    )
    state_dict = {
        "weight": torch.randn(2, 32, dtype=torch.bfloat16),
        "weight_scale": scale,
    }

    compressed = NVFP4E5M3PackedCompressor.compress(state_dict, scheme)
    assert set(compressed) == {"weight_packed", "weight_scale"}
    assert compressed["weight_packed"].shape == (2, 16)
    assert compressed["weight_packed"].dtype == torch.uint8
    assert compressed["weight_scale"].dtype == torch.uint8
    assert torch.equal(compressed["weight_scale"], ue5m3_encode(scale))

    decompressed = NVFP4E5M3PackedCompressor.decompress(compressed, scheme)
    assert decompressed["weight"].shape == state_dict["weight"].shape
    assert decompressed["weight"].dtype == state_dict["weight"].dtype
    assert torch.equal(
        decompressed["weight_scale"],
        ue5m3_decode(compressed["weight_scale"], dtype=torch.bfloat16),
    )


def test_gs_compress_decompress_round_trip():
    scheme = preset_name_to_scheme("NVFP4_E5M3_GS", ["Linear"])
    scale = torch.tensor(
        [[0.5, 1.0], [2.0, 8.0]],
        dtype=torch.float32,
    )
    state_dict = {
        "weight": torch.randn(2, 32, dtype=torch.bfloat16),
        "weight_scale": scale,
        "weight_global_scale": torch.tensor([3.0]),
        "input_global_scale": torch.tensor([4.0]),
    }

    compressed = NVFP4E5M3GSPackedCompressor.compress(state_dict, scheme)
    assert set(compressed) == {
        "weight_packed",
        "weight_scale",
        "weight_global_scale",
        "input_global_scale",
    }
    assert compressed["weight_scale"].dtype == torch.uint8
    assert compressed["weight_global_scale"].dtype == torch.float32

    decompressed = NVFP4E5M3GSPackedCompressor.decompress(compressed, scheme)
    assert decompressed["weight"].shape == state_dict["weight"].shape
    assert decompressed["weight"].dtype == state_dict["weight"].dtype
    assert torch.equal(decompressed["weight_global_scale"], torch.tensor([3.0]))
    assert torch.equal(decompressed["input_global_scale"], torch.tensor([4.0]))
    assert torch.equal(
        decompressed["weight_scale"],
        ue5m3_decode(compressed["weight_scale"], dtype=torch.bfloat16),
    )


def test_no_gs_compress_meta_device():
    scheme = preset_name_to_scheme("NVFP4_E5M3", ["Linear"])
    state_dict = {
        "weight": torch.empty(128, 64, device="meta"),
        "weight_scale": torch.empty(128, 4, device="meta"),
    }

    compressed = NVFP4E5M3PackedCompressor.compress(state_dict, scheme)

    assert compressed["weight_packed"].device.type == "meta"
    assert compressed["weight_packed"].shape == (128, 32)
    assert compressed["weight_packed"].dtype == torch.uint8
    assert compressed["weight_scale"].device.type == "meta"
    assert compressed["weight_scale"].shape == (128, 4)
    assert compressed["weight_scale"].dtype == torch.uint8


def test_initialize_global_scale_policy():
    no_gs = preset_name_to_scheme("NVFP4_E5M3", ["Linear"])
    no_gs_module = torch.nn.Linear(32, 32)
    initialize_module_for_quantization(no_gs_module, no_gs)
    assert hasattr(no_gs_module, "weight_scale")
    assert not hasattr(no_gs_module, "weight_global_scale")
    assert not hasattr(no_gs_module, "input_global_scale")

    gs = preset_name_to_scheme("NVFP4_E5M3_GS", ["Linear"])
    gs_module = torch.nn.Linear(32, 32)
    initialize_module_for_quantization(gs_module, gs)
    assert hasattr(gs_module, "weight_global_scale")
    assert hasattr(gs_module, "input_global_scale")
