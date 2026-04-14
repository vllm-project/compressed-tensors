# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload import update_offload_parameter
from compressed_tensors.quantization.utils import calculate_qparams


# ---------------------------------------------------------------------------
# XPU emulation tests (part 2): TorchFunctionMode device emulation
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--emulate-xpu",
        action="store_true",
        default=False,
        help="Emulate XPU device identity on CUDA hardware via TorchFunctionMode",
    )


def pytest_configure(config):
    """Activate device emulation before test collection (before module imports).

    Three layers of patching:
      1. DeviceRemapMode — intercepts torch.* functions, remaps "xpu" -> "cuda"
      2. Accelerator mock — torch.accelerator.current_accelerator() reports "xpu"
      3. is_accelerator_type patch — accepts both "xpu" and "cuda"

    """
    if not config.getoption("--emulate-xpu"):
        return

    from tests.emulate_device import DeviceRemapMode

    real_type = torch.accelerator.current_accelerator().type  # "cuda"
    fake_type = "xpu"

    # Save originals for cleanup
    config._emulate_orig_current_accelerator = torch.accelerator.current_accelerator
    config._emulate_orig_device_count = torch.accelerator.device_count
    config._emulate_orig_is_available = torch.accelerator.is_available

    # Snapshot real values before mocking
    real_device_count = torch.accelerator.device_count()
    real_is_available = torch.accelerator.is_available()

    # Layer 1: DeviceRemapMode
    mode = DeviceRemapMode(fake_type=fake_type, real_type=real_type)
    mode.__enter__()
    config._emulate_device_remap_mode = mode

    # Layer 2: Mock accelerator identity
    # Use SimpleNamespace for current_accelerator (reports "xpu"), but also
    # mock device_count and is_available to return real values directly —
    # they internally call current_accelerator() and try to hash the result,
    # which fails with SimpleNamespace.
    fake_accel = SimpleNamespace(type=fake_type)
    torch.accelerator.current_accelerator = lambda: fake_accel
    torch.accelerator.device_count = lambda: real_device_count
    torch.accelerator.is_available = lambda: real_is_available

    # Layer 3: Patch is_accelerator_type to accept both types
    import compressed_tensors.utils as _utils

    config._emulate_orig_is_accelerator_type = _utils.is_accelerator_type

    def patched_is_accelerator_type(device_type: str) -> bool:
        return device_type in (fake_type, real_type)

    _utils.is_accelerator_type = patched_is_accelerator_type

    # Also patch base.py's binding since it imported is_accelerator_type directly
    # and captured the original function before pytest_configure ran
    import compressed_tensors.offload.cache.base as _base

    config._emulate_orig_base_is_accelerator_type = _base.is_accelerator_type
    _base.is_accelerator_type = patched_is_accelerator_type


def pytest_unconfigure(config):
    """Tear down device emulation — restore all patched objects."""
    mode = getattr(config, "_emulate_device_remap_mode", None)
    if mode is not None:
        mode.__exit__(None, None, None)

    orig_accel = getattr(config, "_emulate_orig_current_accelerator", None)
    if orig_accel is not None:
        torch.accelerator.current_accelerator = orig_accel
        torch.accelerator.device_count = config._emulate_orig_device_count
        torch.accelerator.is_available = config._emulate_orig_is_available

    orig_is_accel = getattr(config, "_emulate_orig_is_accelerator_type", None)
    if orig_is_accel is not None:
        import compressed_tensors.utils as _utils

        _utils.is_accelerator_type = orig_is_accel

    orig_base_is_accel = getattr(config, "_emulate_orig_base_is_accelerator_type", None)
    if orig_base_is_accel is not None:
        import compressed_tensors.offload.cache.base as _base

        _base.is_accelerator_type = orig_base_is_accel


# ---------------------------------------------------------------------------
# Calibration fixtures
# ---------------------------------------------------------------------------


def _get_dim(dim: int, value: torch.Tensor):
    if isinstance(dim, int):
        dim = [dim]
        dim = set(dim)

    reduce_dims = tuple(idx for idx in range(value.ndim) if idx not in dim)
    return reduce_dims


@pytest.fixture
def mock_per_group_calibration():
    def update_scale_zp(
        module: torch.nn.Module, base_name: str, value: torch.Tensor, group_size: int
    ):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
        args = getattr(quantization_scheme, arg_name, None)

        rows = value.shape[0]
        columns = value.shape[1]
        num_groups = int(ceil(columns / group_size))

        scale = torch.zeros((rows, num_groups), dtype=value.dtype, device=value.device)
        zp_dtype = args.pytorch_dtype()
        zp = torch.zeros((rows, num_groups), dtype=zp_dtype, device=value.device)

        group_sizes = torch.full((num_groups,), group_size, dtype=torch.int)
        end = 0
        for group_index, group_count in enumerate(group_sizes):
            start = end
            end = start + group_count
            dim = _get_dim(
                0,
                value[:, start:end],
            )
            min_val = torch.amin(value, dim=dim, keepdims=True)
            max_val = torch.amax(value, dim=dim, keepdims=True)
            scale_out, zp_out = calculate_qparams(min_val, max_val, args)

            scale[:, group_index] = scale_out.squeeze(1)
            zp[:, group_index] = zp_out.squeeze(1)

        update_offload_parameter(module, f"{base_name}_scale", scale)
        update_offload_parameter(module, f"{base_name}_zero_point", zp)

    return update_scale_zp


@pytest.fixture
def mock_per_channel_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"

        args = getattr(quantization_scheme, arg_name, None)
        dim = _get_dim(0, value)
        min_val = torch.amin(value, dim=dim, keepdims=True)
        max_val = torch.amax(value, dim=dim, keepdims=True)
        scale, zp = calculate_qparams(min_val, max_val, args)
        update_offload_parameter(module, f"{base_name}_scale", scale)
        update_offload_parameter(module, f"{base_name}_zero_point", zp)

    return update_scale_zp


@pytest.fixture
def mock_per_tensor_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
        args = getattr(quantization_scheme, arg_name, None)

        # per tensor quantization just calls calculate_qparams directly
        min_val, max_val = torch.aminmax(value)
        scale, zp = calculate_qparams(min_val, max_val, args)
        update_offload_parameter(module, f"{base_name}_scale", scale)
        update_offload_parameter(module, f"{base_name}_zero_point", zp)

    return update_scale_zp
