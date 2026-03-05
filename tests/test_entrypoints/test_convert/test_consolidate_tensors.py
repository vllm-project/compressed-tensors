# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from compressed_tensors.entrypoints.convert.consolidate import consolidate_tensors
from safetensors.torch import load_file, save_file


class TestConsolidateTensors:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def _create_test_checkpoint(
        self,
        save_dir: str,
        weight_map: dict[str, str],
        create_index: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Helper to create test safetensors files and optionally an index.

        :param save_dir: directory to save files
        :param weight_map: mapping of tensor_name -> file_name
        :param create_index: whether to create an index file
        :return: dict of file_name -> tensors
        """
        # Group tensors by file
        file_tensors = {}
        for tensor_name, file_name in weight_map.items():
            if file_name not in file_tensors:
                file_tensors[file_name] = {}
            # Create random tensors for testing
            file_tensors[file_name][tensor_name] = torch.randn(10, 10)

        # Save each file
        total_size = 0
        for file_name, tensors in file_tensors.items():
            file_path = Path(save_dir) / file_name
            save_file(tensors, file_path)
            total_size += sum(tensor.nbytes for tensor in tensors.values())

        # Optionally create index file
        if create_index:
            index_data = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            index_path = Path(save_dir) / "model.safetensors.index.json"
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)

        return file_tensors

    @pytest.mark.unit
    def test_no_safetensors_files(self, temp_dir):
        """Test that consolidate_tensors handles missing safetensors files gracefully"""
        # Should not raise an error, just log and return
        consolidate_tensors(temp_dir, temp_dir)

    @pytest.mark.unit
    def test_simple_consolidation_in_place(self, temp_dir):
        """Test basic consolidation in-place (same directory)"""
        weight_map = {
            "model.layer1.weight": "model-00001-of-00002.safetensors",
            "model.layer1.bias": "model-00002-of-00002.safetensors",  # Split!
        }

        original_tensors = self._create_test_checkpoint(temp_dir, weight_map, create_index=False)

        # Consolidate in-place (save_directory=None)
        consolidate_tensors(temp_dir, save_directory=None)

        # Verify files were actually updated
        file1_path = Path(temp_dir) / "model-00001-of-00002.safetensors"
        file1_tensors = load_file(file1_path)
        assert "model.layer1.weight" in file1_tensors
        assert "model.layer1.bias" in file1_tensors

        # File2 should not exist since it's now empty
        file2_path = Path(temp_dir) / "model-00002-of-00002.safetensors"
        assert not file2_path.exists()

        # Verify tensor values are preserved
        assert torch.equal(
            file1_tensors["model.layer1.weight"],
            original_tensors["model-00001-of-00002.safetensors"]["model.layer1.weight"],
        )
        assert torch.equal(
            file1_tensors["model.layer1.bias"],
            original_tensors["model-00002-of-00002.safetensors"]["model.layer1.bias"],
        )

    @pytest.mark.unit
    def test_consolidation_to_new_directory(self, temp_dir):
        """Test consolidation to a new output directory with all files copied"""
        weight_map = {
            "model.layer1.weight": "model-00001-of-00002.safetensors",
            "model.layer1.bias": "model-00002-of-00002.safetensors",  # Split!
        }

        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()

        original_tensors = self._create_test_checkpoint(str(input_dir), weight_map, create_index=True)

        # Create additional files that should be copied
        config_file = input_dir / "config.json"
        config_file.write_text('{"model_type": "test"}')
        tokenizer_file = input_dir / "tokenizer.json"
        tokenizer_file.write_text('{"version": "1.0"}')

        # Consolidate to new directory
        consolidate_tensors(input_dir, output_dir)

        # Verify output directory has consolidated files
        file1_path = output_dir / "model-00001-of-00002.safetensors"
        assert file1_path.exists()
        file1_tensors = load_file(file1_path)
        assert "model.layer1.weight" in file1_tensors
        assert "model.layer1.bias" in file1_tensors

        # File2 should not exist in output (empty)
        file2_path = output_dir / "model-00002-of-00002.safetensors"
        assert not file2_path.exists()

        # Original input files should remain unchanged
        input_file1 = input_dir / "model-00001-of-00002.safetensors"
        input_file2 = input_dir / "model-00002-of-00002.safetensors"
        assert input_file1.exists()
        assert input_file2.exists()

        # Verify index file was copied and updated
        index_path = output_dir / "model.safetensors.index.json"
        assert index_path.exists()

        # Verify all other files were copied
        assert (output_dir / "config.json").exists()
        assert (output_dir / "config.json").read_text() == '{"model_type": "test"}'
        assert (output_dir / "tokenizer.json").exists()
        assert (output_dir / "tokenizer.json").read_text() == '{"version": "1.0"}'

        # Verify tensor values are preserved
        assert torch.equal(
            file1_tensors["model.layer1.weight"],
            original_tensors["model-00001-of-00002.safetensors"]["model.layer1.weight"],
        )
        assert torch.equal(
            file1_tensors["model.layer1.bias"],
            original_tensors["model-00002-of-00002.safetensors"]["model.layer1.bias"],
        )

    @pytest.mark.unit
    def test_consolidation_same_directory_explicit(self, temp_dir):
        """Test that explicitly passing same directory as both args works as in-place"""
        weight_map = {
            "model.layer1.weight": "model-00001-of-00002.safetensors",
            "model.layer1.bias": "model-00002-of-00002.safetensors",
        }

        self._create_test_checkpoint(temp_dir, weight_map, create_index=False)

        # Pass same directory for both parameters
        consolidate_tensors(temp_dir, temp_dir)

        # Should work like in-place consolidation
        file1_path = Path(temp_dir) / "model-00001-of-00002.safetensors"
        file1_tensors = load_file(file1_path)
        assert "model.layer1.weight" in file1_tensors
        assert "model.layer1.bias" in file1_tensors

        # Empty file should be removed
        file2_path = Path(temp_dir) / "model-00002-of-00002.safetensors"
        assert not file2_path.exists()

    @pytest.mark.unit
    def test_complex_module_names(self, temp_dir):
        """Test consolidation with complex module names like MoE experts"""
        weight_map = {
            "model.layers.60.mlp.experts.84.up_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.60.mlp.experts.84.up_proj.weight_scale_inv": "model-00002-of-00002.safetensors",
        }

        self._create_test_checkpoint(temp_dir, weight_map, create_index=False)
        consolidate_tensors(temp_dir, save_directory=None)

        # Verify both tensors consolidated to first file
        file1_tensors = load_file(
            Path(temp_dir) / "model-00001-of-00002.safetensors"
        )
        assert "model.layers.60.mlp.experts.84.up_proj.weight" in file1_tensors
        assert "model.layers.60.mlp.experts.84.up_proj.weight_scale_inv" in file1_tensors

    @pytest.mark.unit
    def test_adjacent_file_consolidation_only(self, temp_dir):
        """Test that only adjacent files are consolidated"""
        weight_map = {
            "model.layer1.weight": "model-00001-of-00003.safetensors",
            "model.layer1.bias": "model-00002-of-00003.safetensors",
            "model.layer2.weight": "model-00003-of-00003.safetensors",
        }

        self._create_test_checkpoint(temp_dir, weight_map, create_index=False)
        consolidate_tensors(temp_dir, save_directory=None)

        # File 1 should get file 2's tensor (adjacent)
        file1_tensors = load_file(
            Path(temp_dir) / "model-00001-of-00003.safetensors"
        )
        assert "model.layer1.weight" in file1_tensors
        assert "model.layer1.bias" in file1_tensors

        # File 2 should be removed (empty after moving tensor to file 1)
        assert not (Path(temp_dir) / "model-00002-of-00003.safetensors").exists()

        # File 3 should remain unchanged (different module)
        file3_tensors = load_file(
            Path(temp_dir) / "model-00003-of-00003.safetensors"
        )
        assert "model.layer2.weight" in file3_tensors

    @pytest.mark.unit
    def test_consolidation_with_index_update(self, temp_dir):
        """Test that index file is correctly updated when it exists"""
        weight_map = {
            "model.layer1.weight": "model-00001-of-00002.safetensors",
            "model.layer1.bias": "model-00002-of-00002.safetensors",
        }

        original_tensors = self._create_test_checkpoint(temp_dir, weight_map, create_index=True)
        consolidate_tensors(temp_dir, save_directory=None)

        # Verify index was updated
        index_path = Path(temp_dir) / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            updated_index = json.load(f)

        # Both tensors should now be in file 1
        assert (
            updated_index["weight_map"]["model.layer1.weight"]
            == "model-00001-of-00002.safetensors"
        )
        assert (
            updated_index["weight_map"]["model.layer1.bias"]
            == "model-00001-of-00002.safetensors"
        )

        # Verify total size is correct
        expected_size = sum(
            tensor.nbytes
            for file_tensors in original_tensors.values()
            for tensor in file_tensors.values()
        )
        assert updated_index["metadata"]["total_size"] == expected_size

    @pytest.mark.unit
    def test_non_local_path_requires_save_directory(self):
        """Test that non-local path without save_directory raises an error"""
        # Use a path that doesn't exist locally
        non_local_path = "some-org/some-model"

        with pytest.raises(ValueError, match="save_directory is required"):
            consolidate_tensors(non_local_path, save_directory=None)
