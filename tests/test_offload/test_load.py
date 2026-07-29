# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import MagicMock, patch

import compressed_tensors.offload.load as load_module
import pytest
import torch
from compressed_tensors.distributed import utils as dist_utils
from compressed_tensors.distributed.utils import is_distributed
from compressed_tensors.offload import (
    disable_onloading,
    from_accelerate,
    get_offloaded_device,
)
from compressed_tensors.offload.convert import to_accelerate
from compressed_tensors.offload.convert.from_accelerate import _infer_module_device
from compressed_tensors.offload.load import load_offloaded_model
from compressed_tensors.offload.utils import as_single_threaded
from tests.test_offload.conftest import (
    assert_device_equal,
    skip_if_mps_device,
    torchrun,
)
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM


acclerate = pytest.importorskip("accelerate")


accelerator_device = torch.accelerator.current_accelerator()
TEST_PARAMETERS = [
    (
        "auto",
        {0: 596049920, "cpu": 1e15},  # force cpu offload for testing
        accelerator_device,
        torch.device("cpu"),
    ),
    (
        accelerator_device.type,
        None,
        accelerator_device,
        accelerator_device,
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
def test_load(device_map, max_memory, first, second, tmp_path):
    with load_offloaded_model(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map=device_map,
            max_memory=max_memory,
            dtype=torch.bfloat16,
            offload_folder=str(tmp_path / "disk_offload"),
        )

    for layer_index in range(0, 8):
        module = model.get_submodule(f"model.layers.{layer_index}.self_attn.q_proj")
        assert_device_equal(get_offloaded_device(module), first)

    for layer_index in range(8, 28):
        module = model.get_submodule(f"model.layers.{layer_index}.self_attn.q_proj")
        assert_device_equal(get_offloaded_device(module), second)

    with disable_onloading():
        state_dict = model.state_dict(keep_vars=True)

    to_accelerate(model)

    for layer_index in range(0, 8):
        module = model.get_submodule(f"model.layers.{layer_index}.self_attn.q_proj")
        assert_device_equal(_get_accelerate_offloaded_device(module), first)

    for layer_index in range(8, 28):
        module = model.get_submodule(f"model.layers.{layer_index}.self_attn.q_proj")
        assert_device_equal(_get_accelerate_offloaded_device(module), second)

    model.save_pretrained(tmp_path / "save_path")

    from_accelerate(model)

    # TODO: accelerate's disk onloading implementation does not keep consistent meta
    # tensors, :. tensor pointers change and cannot be converted back properly
    if second != "disk":
        with disable_onloading():
            assert model.state_dict(keep_vars=True) == state_dict


@pytest.mark.integration
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_load_dist(tmp_path):
    for parameters in TEST_PARAMETERS:
        test_load(*parameters, tmp_path=tmp_path)


def _get_accelerate_offloaded_device(module: torch.nn.Module) -> str | None:
    device = _infer_module_device(module)
    if device == torch.device("meta"):
        return "disk"

    return device


@pytest.mark.unit
@skip_if_mps_device
@patch("compressed_tensors.offload.load.from_accelerate")
def test_patch_forwards_positional_args(mock_from_accelerate):
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

    with load_offloaded_model(FakeModel, extra_cpu_mem=0):
        FakeModel.from_pretrained("org/model", device_map="cpu", torch_dtype="auto")

    assert received["cls"] is FakeModel
    assert received["path"] == "org/model"
    assert received["kwargs"]["device_map"] == "cpu"
    assert received["kwargs"]["torch_dtype"] == "auto"


@pytest.mark.unit
def test_mmap_cap_reduces_shared_memory():
    """Tight mmap limit reduces _get_shared_memory return value."""
    with (
        patch.object(load_module, "_get_max_map_count", return_value=100),
        patch.object(load_module, "_get_current_map_count", return_value=50),
    ):
        # 500 tensors, 1024 bytes each = 512KB total
        # Available maps = 100 - 50 = 50
        # Avg tensor = 512000 / 500 = 1024 bytes
        # mmap budget = 50 * 1024 = 51200
        result = load_module._get_shared_memory(
            num_tensors=500, total_model_bytes=512000
        )
        assert result == 51200


@pytest.mark.unit
def test_mmap_cap_no_reduction_when_limit_high():
    """When mmap limit is very high, byte capacity is the bottleneck."""
    with (
        patch.object(load_module, "_get_max_map_count", return_value=10_000_000),
        patch.object(load_module, "_get_current_map_count", return_value=500),
    ):
        result = load_module._get_shared_memory(
            num_tensors=5000, total_model_bytes=5_000_000_000
        )
        # Should be the /dev/shm byte size, not reduced by mmap
        import shutil

        if os.path.exists("/dev/shm"):
            expected = shutil.disk_usage("/dev/shm").total
            assert result == expected


@pytest.mark.unit
def test_mmap_cap_graceful_on_non_linux():
    """When /proc files aren't available, skip the mmap cap entirely."""
    with patch.object(load_module, "_get_max_map_count", return_value=None):
        # Should not crash, should return full /dev/shm size
        result = load_module._get_shared_memory(
            num_tensors=5000, total_model_bytes=5_000_000_000
        )
        assert result > 0


@pytest.mark.unit
def test_mmap_cap_skipped_without_tensor_info():
    """When no tensor info provided, behave like before (bytes only)."""
    result_no_info = load_module._get_shared_memory()
    result_zero = load_module._get_shared_memory(num_tensors=0, total_model_bytes=0)
    assert result_no_info == result_zero


@pytest.mark.integration
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_load_dist_estimate_tensor_count(tmp_path):
    """Loading with default max_memory triggers _estimate_tensor_count without hang."""
    with load_offloaded_model(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained(
            "inference-optimization/Llama-3.2-1B-Instruct-FP8-Block",
            device_map="auto",
            dtype=torch.bfloat16,
        )
    assert model is not None


@pytest.mark.unit
def test_as_single_threaded_toggles_is_distributed():
    """as_single_threaded suppresses is_distributed and restores on exit."""
    with patch.object(dist_utils, "_force_single_threaded", False), patch(
        "torch.distributed.is_available", return_value=True
    ), patch("torch.distributed.is_initialized", return_value=True):
        assert is_distributed() is True

        with as_single_threaded():
            assert is_distributed() is False

        assert is_distributed() is True
