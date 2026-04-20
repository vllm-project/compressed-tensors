# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
XPU emulation test (part 1): Mock routing tests for XPU device type.

Verifies that routing functions handle ``"xpu"`` correctly by mocking
``torch.accelerator`` — no real tensor operations, no GPU required.

These tests are skipped when ``--emulate-xpu`` is active because the global
``is_accelerator_type`` patch conflicts with the per-test monkeypatch.
"""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.device import DeviceCache
from compressed_tensors.offload.convert.helpers import norm_device
from compressed_tensors.utils import is_accelerator_type


@pytest.fixture
def mock_xpu_accelerator(monkeypatch):
    """Mock torch.accelerator to report XPU as the current device."""
    fake = SimpleNamespace(type="xpu")
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: fake)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 1)


skipif_emulate_xpu = pytest.mark.skipif(
    "config.getoption('--emulate-xpu', default=False)",
    reason="Option 1 routing tests conflict with --emulate-xpu global patches",
)


@pytest.mark.unit
@skipif_emulate_xpu
class TestXpuRouting:
    """Verify that routing functions correctly handle 'xpu' as the accelerator type.

    These tests mock torch.accelerator without real tensor operations —
    they validate the decision logic that changed in the torch.accelerator migration.
    """

    def test_is_accelerator_type_xpu(self, mock_xpu_accelerator):
        assert is_accelerator_type("xpu") is True
        assert is_accelerator_type("cuda") is False
        assert is_accelerator_type("cpu") is False

    def test_is_accelerator_type_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
        assert is_accelerator_type("xpu") is False

    def test_cache_routes_device_cache_for_xpu(self, mock_xpu_accelerator):
        cache_cls = OffloadCache.cls_from_device(torch.device("xpu", 0))
        assert cache_cls is DeviceCache

    def test_norm_device_resolves_xpu_to_index_0(self, mock_xpu_accelerator):
        result = norm_device("xpu")
        assert result == torch.device("xpu", 0)

    def test_norm_device_preserves_xpu_with_index(self, mock_xpu_accelerator):
        result = norm_device(torch.device("xpu", 0))
        assert result == torch.device("xpu", 0)

    def test_norm_device_cpu_unaffected(self, mock_xpu_accelerator):
        result = norm_device("cpu")
        assert result == torch.device("cpu")

    def test_get_safe_open_device_xpu(self, mock_xpu_accelerator):
        from compressed_tensors.offload.cache.disk import _get_safe_open_device

        # bare "xpu" → current device index (0)
        result = _get_safe_open_device(torch.device("xpu"))
        assert result == "xpu:0"

    def test_get_safe_open_device_xpu_with_index(self, mock_xpu_accelerator):
        from compressed_tensors.offload.cache.disk import _get_safe_open_device

        result = _get_safe_open_device(torch.device("xpu", 3))
        assert result == "xpu:3"

    def test_get_safe_open_device_cuda_returns_string(self, monkeypatch):
        from compressed_tensors.offload.cache.disk import _get_safe_open_device

        fake = SimpleNamespace(type="cuda")
        monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: fake)
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)

        assert _get_safe_open_device(torch.device("cuda")) == "cuda:2"
        assert _get_safe_open_device(torch.device("cuda", 5)) == "cuda:5"

    def test_get_safe_open_device_cpu(self, mock_xpu_accelerator):
        from compressed_tensors.offload.cache.disk import _get_safe_open_device

        result = _get_safe_open_device(torch.device("cpu"))
        assert result == "cpu"
