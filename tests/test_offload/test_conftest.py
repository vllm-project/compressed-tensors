# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from tests.test_offload.conftest import assert_device_equal


@pytest.mark.unit
def test_assert_device_equal_without_accelerator(monkeypatch):
    """CPU-only runs compare CPU devices without querying an accelerator index."""
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: None)
    monkeypatch.setattr(
        torch.accelerator,
        "current_device_index",
        lambda: pytest.fail("CPU-only comparison must not query an accelerator index"),
    )

    assert_device_equal(torch.device("cpu"), torch.device("cpu"))
