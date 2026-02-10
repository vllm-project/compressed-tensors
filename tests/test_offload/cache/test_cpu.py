# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from tests.test_offload.cache.helpers import (
    _test_delete,
    _test_disable_offloading,
    _test_disable_onloading,
    _test_garbage_collect,
    _test_offload,
    _test_onload,
    _test_onloading,
    _test_shared_attributes,
    _test_tensor_subclass,
)
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda")
OFFLOAD_DEVICE = torch.device("cpu")


@pytest.mark.unit
@requires_gpu
def test_delete():
    _test_delete(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_disable_offloading():
    _test_disable_offloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_disable_onloading():
    _test_disable_onloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_garbage_collect():
    _test_garbage_collect(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_offload():
    _test_offload(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
@requires_gpu
def test_onload():
    _test_onload(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_onloading():
    _test_onloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_shared_attributes():
    _test_shared_attributes(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_tensor_subclass():
    _test_tensor_subclass(OFFLOAD_DEVICE, ONLOAD_DEVICE)
