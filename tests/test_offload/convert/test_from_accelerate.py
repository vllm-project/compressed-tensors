# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import (
    disable_onloading,
    from_accelerate,
    is_rank0,
    load_offloaded_model,
)
from compressed_tensors.offload.cache import CPUCache, DeviceCache, DiskCache
from compressed_tensors.offload.convert.from_accelerate import (
    remove_accelerate_from_module,
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM


acclerate = pytest.importorskip("accelerate")


@pytest.mark.unit
@requires_gpu
def test_remove_accelerate_from_module_device(cuda_device):
    # there"s no way to force accelerate to "offload" to cuda. Instead, it just
    # stays on cuda with no hooks
    linear = torch.nn.Linear(5, 5, device="cuda:0")
    assert remove_accelerate_from_module(linear) == (cuda_device, cuda_device, None)
    assert not hasattr(linear, "_hf_hook")

    # test idempotency
    assert remove_accelerate_from_module(linear) == (cuda_device, cuda_device, None)
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
def test_remove_accelerate_from_module_cpu(cuda_device):
    from accelerate.big_modeling import dispatch_model

    linear = torch.nn.Linear(5, 5)
    dispatch_model(
        linear,
        {"": "cpu"},
        main_device="cuda",
        state_dict=linear.state_dict(),
        force_hooks=True,
    )
    assert remove_accelerate_from_module(linear) == (
        cuda_device,
        torch.device("cpu"),
        None,
    )
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_remove_accelerate_from_module_disk(cuda_device, tmp_path):
    # `disk_offload` is a super buggy function, and not reflective of real dispatches
    # `dispatch_model` is also super buggy, and requires at least one cpu device
    from accelerate.big_modeling import dispatch_model

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    linear = torch.nn.Linear(5, 5)
    model = torch.nn.Sequential(linear)
    dispatch_model(
        model,
        {"0": "disk", "fake_module": "cpu"},
        main_device="cuda",
        force_hooks=True,
        offload_dir=offload_dir,
    )
    assert remove_accelerate_from_module(linear) == (cuda_device, "disk", offload_dir)
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
def test_from_accelerate(cuda_device, tmp_path):
    from accelerate.big_modeling import dispatch_model

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )
    if is_rank0():
        dispatch_model(
            model,
            {"0": 0, "1": "cpu", "2": "disk"},
            main_device=str(cuda_device),
            force_hooks=True,
            offload_dir=offload_dir,
        )
    else:
        model.to("meta")

    device_map, _offload_dir = from_accelerate(model)

    # cuda is index agnostic when distributed
    assert device_map == {
        "": (None, None),
        "0": (cuda_device, cuda_device),
        "1": (cuda_device, torch.device("cpu")),
        "2": (cuda_device, "disk"),
    }
    if is_rank0():
        assert _offload_dir == offload_dir
    assert isinstance(model[0]._parameters, DeviceCache)
    assert isinstance(model[1]._parameters, CPUCache)
    assert isinstance(model[2]._parameters, DiskCache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_from_accelerate_dist(cuda_device, tmp_path):
    test_from_accelerate(cuda_device, tmp_path)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
@torch.no_grad()
def test_dist_disk_safetensors_update(tmp_path):
    offload_folder = tmp_path / "offload_folder"
    os.makedirs(offload_folder, exist_ok=True)

    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            dtype="auto",
            device_map="auto_offload",
            max_memory={"cpu": 6e8},
            offload_folder=str(offload_folder),
        )

        # Get the model checkpoint files and hash them
        if dist.get_rank() == 0:
            checkpoint_files = {}
            for file_path in Path(offload_folder).glob("*.safetensors"):
                if not file_path.name.startswith(DiskCache._new_file_prefix):
                    with open(file_path, "rb") as f:
                        checkpoint_files[file_path.name] = hashlib.sha256(
                            f.read()
                        ).hexdigest()

        dist.barrier()

        # Each rank updates a different module's tensor
        rank_0_module = model.model.layers[-1].self_attn.q_proj
        rank_1_module = model.model.layers[-1].self_attn.k_proj
        rank = dist.get_rank()
        if rank == 0:
            rank_0_module.weight *= 0
        elif rank == 1:
            rank_1_module.weight *= 0
            rank_1_module.weight += 1
        dist.barrier()

        # Check that onloaded values are updated across ranks
        assert torch.all(rank_0_module.weight == 0)
        assert torch.all(rank_1_module.weight == 1)

        # Compare model checkpoint files and make sure they're unchanged
        if dist.get_rank() == 0:
            for file_name, original_hash in checkpoint_files.items():
                file_path = offload_folder / file_name
                with open(file_path, "rb") as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                assert current_hash == original_hash

        # Check that the files exist and are not symlinks
        with disable_onloading():
            q_file_path = DiskCache.index[rank_0_module.weight]["safetensors_file"]
            k_file_path = DiskCache.index[rank_1_module.weight]["safetensors_file"]

        if dist.get_rank() == 0:
            assert os.path.exists(q_file_path)
            assert not os.path.islink(q_file_path)
        if dist.get_rank() == 1:
            assert os.path.exists(k_file_path)
            assert not os.path.islink(k_file_path)

        # Delete the parameters
        delattr(rank_0_module, "weight")
        delattr(rank_1_module, "weight")

        # Wait for all ranks to complete deletion
        dist.barrier()

        # Check that the new files were deleted
        assert not os.path.exists(q_file_path)
        assert not os.path.exists(k_file_path)

        # Compare model checkpoint files again and make sure they're still unchanged
        if dist.get_rank() == 0:
            for file_name, original_hash in checkpoint_files.items():
                file_path = offload_folder / file_name
                with open(file_path, "rb") as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                assert current_hash == original_hash
