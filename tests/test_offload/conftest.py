# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from functools import wraps
from types import FunctionType
from typing import Any, Callable, Literal, Optional

import pytest
import torch
from compressed_tensors.offload.utils import send_tensors


accelerator_device = torch.accelerator.current_accelerator()

skip_if_mps_device = pytest.mark.skipif(
    accelerator_device.type == "mps",
    reason="[Known issue] https://github.com/pytorch/pytorch/issues/167447",
)


def assert_device_equal(
    device_a: torch.device | Literal["disk"],
    device_b: torch.device | Literal["disk"],
):
    if device_a == "disk":
        device_a = torch.device("meta")
    if device_b == "disk":
        device_b = torch.device("meta")

    cur_index = torch.accelerator.current_device_index()
    a_index = cur_index if device_a.index is None else device_a.index
    b_index = cur_index if device_b.index is None else device_b.index

    # Handle device emulation: when --emulate-xpu is active, tensors created
    # on "xpu" actually live on the real accelerator, so their .device reports
    # the real type. Normalize device types: if one matches the fake type and
    # the other matches the real type, treat them as equal.
    accel = torch.accelerator.current_accelerator()
    fake_type = accel.type
    real_type = getattr(accel, "_real_type", None)

    a_type = device_a.type
    b_type = device_b.type

    # If emulation is active, normalize: fake_type and real_type are equivalent
    if real_type is not None:
        if a_type == real_type:
            a_type = fake_type
        if b_type == real_type:
            b_type = fake_type

    assert a_type == b_type and a_index == b_index


def assert_tensor_equal(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    device: Optional[torch.device | str] = None,
):
    if device is not None:
        tensor_b = send_tensors(tensor_b, "meta" if device == "disk" else device)

    assert tensor_a.__class__ == tensor_b.__class__
    assert tensor_a.requires_grad == tensor_b.requires_grad
    assert tensor_a.__dict__ == tensor_b.__dict__

    if tensor_a.is_meta or tensor_b.is_meta:
        assert (
            tensor_a.device == tensor_b.device
            and tensor_a.shape == tensor_b.shape
            and tensor_a.dtype == tensor_b.dtype
        )
    else:
        assert torch.equal(tensor_a, tensor_b)


def torchrun(
    world_size: int = 1, init_dist: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Pytest decorator to run a test within parallel torchrun subprocesses.

    This decorator automatically spawns torchrun when the test is run with regular
    pytest.
    When running under torchrun (detected via TORCHELASTIC_RUN_ID env var), it
    optionally initializes the distributed process group before running the test.

    Usage:
        @pytest.mark.unit
        @requires_gpu(2)
        @torchrun(world_size=2, init_dist=True)
        def test_distributed_feature():
            # Distributed already initialized
            ...

        @torchrun(world_size=2)  # init_dist=False by default
        def test_custom_init():
            # Handle your own distributed setup
            from compressed_tensors.distributed import init_dist
            init_dist()
            ...

    :param world_size: number of ranks to spawn
    :param init_dist: whether to automatically call init_dist() (default: False)
    """

    def decorator(func: FunctionType):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # We're running in a torchrun subprocess: optionally init then run the test
            if "TORCHELASTIC_RUN_ID" in os.environ:
                if init_dist:
                    from compressed_tensors.distributed import init_dist as _init_dist

                    _init_dist()
                return func(*args, **kwargs)

            # First time calling in the main process:
            # trigger torchrun with this function as the pytest target
            else:
                module = sys.modules.get(func.__module__)
                if module is None:
                    raise RuntimeError(
                        f"Can't find module {func.__module__} for func {func.__name__}"
                    )
                file_path = module.__file__
                if file_path is None:
                    raise RuntimeError(
                        f"Module {func.__module__} has no __file__ attribute"
                    )
                func_name = func.__name__

                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node",
                    str(world_size),
                    "--log-dir",
                    "/tmp/torchrun-logs",
                    "--tee",
                    "3",
                    "--role",
                    "torchrun",
                    "-m",
                    "pytest",
                    f"{file_path}::{func_name}",
                    "-sx",
                ]

                proc = subprocess.run(cmd)
                assert proc.returncode == 0

        return wrapper

    return decorator


@pytest.fixture()
def accel_device():
    accel_type = torch.accelerator.current_accelerator().type
    return (
        torch.device(accel_type)
        if "TORCHELASTIC_RUN_ID" in os.environ
        else torch.device(accel_type, 0)
    )


@pytest.fixture
def offload_folder(tmp_path) -> str:
    """Create an offload folder on rank 0 and broadcast to all ranks."""
    offload_path = tmp_path / "offload_folder"

    # Create directory on rank 0
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs(offload_path, exist_ok=True)

    # If distributed, broadcast the path from rank 0 to all ranks
    if torch.distributed.is_initialized():
        broadcast_object = [str(offload_path)]
        torch.distributed.broadcast_object_list(broadcast_object, src=0)
        offload_path = broadcast_object[0]

    return offload_path
