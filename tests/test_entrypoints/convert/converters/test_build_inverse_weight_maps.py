# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from compressed_tensors.entrypoints.convert import (
    FP8BlockDequantizer,
    build_inverse_weight_maps,
)
from safetensors.torch import load_file, save_file
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
)


@pytest.mark.unit
def test_build_inverse_weight_maps(tmp_path):
    """
    Test that reindex_checkpoint correctly moves tensors across files
    so that weight and weight_scale_inv end up in the same file.
    """
    # Create dummy checkpoint with weights split across files
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # File 1: has layer0.weight but NOT layer0.weight_scale_inv
    file1_tensors = {
        "embed_tokens.weight": torch.randn(128, 128, dtype=torch.float32),
        "layer0.weight": torch.randn(128, 128, dtype=torch.float32).to(
            torch.float8_e4m3fn
        ),
        "layer1.weight_scale_inv": torch.randn(1, 1, dtype=torch.float32),
    }
    file1_path = model_dir / "model-00001-of-00002.safetensors"
    save_file(file1_tensors, str(file1_path))

    # File 2: has layer0.weight_scale_inv and layer1.weight_scale_inv
    file2_tensors = {
        "layer0.weight_scale_inv": torch.randn(1, 1, dtype=torch.float32),
        "layer1.weight": torch.randn(128, 128, dtype=torch.float32).to(
            torch.float8_e4m3fn
        ),
        "layer2.weight": torch.randn(128, 128, dtype=torch.float32).to(
            torch.float8_e4m3fn
        ),
        "layer2.weight_scale_inv": torch.randn(1, 1, dtype=torch.float32),
        "lm_head.weight": torch.randn(128, 128, dtype=torch.float32),
    }
    file2_path = model_dir / "model-00002-of-00002.safetensors"
    save_file(file2_tensors, str(file2_path))

    # Create index file
    weight_map = {
        "embed_tokens.weight": "model-00001-of-00002.safetensors",
        "layer0.weight": "model-00001-of-00002.safetensors",
        "layer1.weight": "model-00002-of-00002.safetensors",
        "layer0.weight_scale_inv": "model-00002-of-00002.safetensors",
        "layer1.weight_scale_inv": "model-00001-of-00002.safetensors",
        "layer2.weight": "model-00002-of-00002.safetensors",
        "layer2.weight_scale_inv": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }

    index_data = {
        "metadata": {
            "total_size": sum(
                t.numel() * t.element_size()
                for tensors in [file1_tensors, file2_tensors]
                for t in tensors.values()
            )
        },
        "weight_map": weight_map,
    }
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f)

    # Create config.json (required by get_checkpoint_files)
    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({"model_type": "test"}, f)

    converter = FP8BlockDequantizer(targets=[r"re:.*layer\d.*"])

    inverse_weight_maps = build_inverse_weight_maps(
        weight_map=weight_map,
        model_files=get_checkpoint_files(model_dir),
        converters=[converter],
    )

    for file_name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        assert (
            file_name in inverse_weight_maps
        ), f"File {file_name} missing in inverse_weight_maps"

    seen_weight_names = set()
    for inverse_weight_map in inverse_weight_maps.values():
        for weight_names in inverse_weight_map.values():
            for weight_name in weight_names:
                assert (
                    weight_name not in seen_weight_names
                ), f"duplicate weight {weight_name} found"
                seen_weight_names.add(weight_name)

    all_weight_names = set(weight_map.keys())
    assert (
        seen_weight_names >= all_weight_names
    ), f"Some weights are missing, {all_weight_names-seen_weight_names}"
    assert (
        all_weight_names >= seen_weight_names
    ), f"Some weights are added that are not included, {seen_weight_names-all_weight_names}"
