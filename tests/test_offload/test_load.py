# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch
from compressed_tensors.offload import get_offloaded_device
from compressed_tensors.offload.convert import to_accelerate
from compressed_tensors.offload.convert.from_accelerate import _infer_module_device
from compressed_tensors.offload.load import load_offloaded_model, patch_from_pretrained
from tests.test_offload.conftest import assert_device_equal, torchrun
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM


acclerate = pytest.importorskip("accelerate")


TEST_PARAMETERS = [
    (
        "auto",
        {0: 596049920, "cpu": 1e15},  # force cpu offload for testing
        torch.device("cuda"),
        torch.device("cpu"),
    ),
    (
        "cuda",
        None,
        torch.device("cuda"),
        torch.device("cuda"),
    ),
    (
        "cpu",
        None,
        torch.device("cpu"),
        torch.device("cpu"),
    ),
    (
        "auto_offload",
        {"cpu": 596049920},  # force disk offload for testing
        torch.device("cpu"),
        "disk",
    ),
]


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize("device_map,max_memory,first,second", TEST_PARAMETERS)
def test_load(device_map, max_memory, first, second):
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map=device_map,
            max_memory=max_memory,
            dtype=torch.bfloat16,
        )

    for layer_index in range(0, 8):
        module = model.get_submodule(f"model.layers.{layer_index}.self_attn.q_proj")
        assert_device_equal(get_offloaded_device(module), first)

    for layer_index in range(8, 28):
        module = model.get_submodule(f"model.layers.{layer_index}.self_attn.q_proj")
        assert_device_equal(get_offloaded_device(module), second)

    to_accelerate(model)

    for layer_index in range(0, 8):
        module = model.get_submodule(f"model.layers.{layer_index}.self_attn.q_proj")
        assert_device_equal(_get_accelerate_offloaded_device(module), first)

    for layer_index in range(8, 28):
        module = model.get_submodule(f"model.layers.{layer_index}.self_attn.q_proj")
        assert_device_equal(_get_accelerate_offloaded_device(module), second)


@pytest.mark.integration
@requires_gpu(2)
@torchrun(world_size=2)
def test_load_dist():
    for parameters in TEST_PARAMETERS:
        test_load(*parameters)


def _get_accelerate_offloaded_device(module: torch.nn.Module) -> str | None:
    device = _infer_module_device(module)
    if device == torch.device("meta"):
        return "disk"

    return device


@pytest.mark.unit
@patch("compressed_tensors.offload.load.from_accelerate")
@patch("compressed_tensors.offload.load.is_rank0", return_value=True)
@patch("compressed_tensors.offload.load.is_distributed", return_value=False)
def test_patch_forwards_positional_args(
    mock_distributed, mock_rank0, mock_from_accelerate
):
    """Regression: positional args must be forwarded without rebinding to cls."""
    received = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            received["cls"] = cls
            received["path"] = pretrained_model_name_or_path
            received["model_args"] = model_args
            received["kwargs"] = kwargs
            return MagicMock()

    with patch_from_pretrained(FakeModel):
        FakeModel.from_pretrained(
            "org/model-name", device_map="cpu", torch_dtype="auto"
        )

    assert received["cls"] is FakeModel
    assert received["path"] == "org/model-name"
    assert received["kwargs"]["device_map"] == "cpu"
    assert received["kwargs"]["torch_dtype"] == "auto"
