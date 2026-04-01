# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from compressed_tensors.entrypoints.convert import (
    FP8BlockDequantizer,
    reindex_checkpoint,
)
from safetensors.torch import load_file, save_file


@pytest.mark.unit
def test_reindex_checkpoint(tmp_path):
    """
    Test that reindex_checkpoint correctly moves tensors across files
    so that weight and weight_scale_inv end up in the same file.
    """
    # Create dummy checkpoint with weights split across files
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # File 1: has layer0.weight but NOT layer0.weight_scale_inv
    file1_tensors = {
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
    }
    file2_path = model_dir / "model-00002-of-00002.safetensors"
    save_file(file2_tensors, str(file2_path))

    # Create index file
    index_data = {
        "metadata": {
            "total_size": sum(
                t.numel() * t.element_size()
                for tensors in [file1_tensors, file2_tensors]
                for t in tensors.values()
            )
        },
        "weight_map": {
            "layer0.weight": "model-00001-of-00002.safetensors",
            "layer1.weight": "model-00002-of-00002.safetensors",
            "layer0.weight_scale_inv": "model-00002-of-00002.safetensors",
            "layer1.weight_scale_inv": "model-00001-of-00002.safetensors",
            "layer2.weight": "model-00002-of-00002.safetensors",
            "layer2.weight_scale_inv": "model-00002-of-00002.safetensors",
        },
    }
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f)

    # Create config.json (required by get_checkpoint_files)
    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({"model_type": "test"}, f)

    converter = FP8BlockDequantizer(targets=[r"re:.*layer\d.*"])

    # Run reindex_checkpoint
    save_dir = tmp_path / "reindexed"
    reindex_checkpoint(
        model_stub=str(model_dir),
        save_directory=str(save_dir),
        get_unmatched_names=converter.get_unmatched_names,
        num_workers=1,
    )

    # Verify the reindexed checkpoint
    # Load the new index file
    new_index_path = save_dir / "model.safetensors.index.json"
    assert new_index_path.exists(), "Index file should exist in save directory"

    with open(new_index_path, "r") as f:
        new_index = json.load(f)

    new_weight_map = new_index["weight_map"]

    # Check that weight and weight_scale_inv are in the same file for each layer
    for layer_idx in range(3):
        weight_file = new_weight_map.get(f"layer{layer_idx}.weight")
        scale_inv_file = new_weight_map.get(f"layer{layer_idx}.weight_scale_inv")

        assert (
            weight_file is not None
        ), f"layer{layer_idx}.weight should be in weight_map"
        assert (
            scale_inv_file is not None
        ), f"layer{layer_idx}.weight_scale_inv should be in weight_map"
        assert (
            weight_file == scale_inv_file
        ), f"layer{layer_idx} weight and scale_inv should be in same file"

    # Verify that actual tensors exist in the files
    for layer_idx in range(3):
        file_name = new_weight_map[f"layer{layer_idx}.weight"]
        file_path = save_dir / file_name
        assert file_path.exists(), f"Safetensors file {file_name} should exist"

        tensors = load_file(str(file_path))
        assert (
            f"layer{layer_idx}.weight" in tensors
        ), f"layer{layer_idx}.weight should be in {file_name}"
        assert (
            f"layer{layer_idx}.weight_scale_inv" in tensors
        ), f"layer{layer_idx}.weight_scale_inv should be in {file_name}"

    # Verify config.json was copied
    assert (save_dir / "config.json").exists(), "config.json should be copied"
