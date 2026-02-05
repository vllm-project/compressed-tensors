# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch
from compressed_tensors.offload.cache.disk import DiskCache
from safetensors import safe_open
from tests.test_offload.cache.helpers import (
    _test_delete,
    _test_disable_offloading,
    _test_disable_onloading,
    _test_garbage_collect,
    _test_offload,
    _test_onload,
    _test_onloading,
    _test_shared_attributes,
)
from tests.test_offload.conftest import assert_tensor_equal
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda")
OFFLOAD_DEVICE = "disk"


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
def test_files(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    # initial write
    DiskCache.index = {}
    cache = DiskCache("cpu", offload_dir=str(offload_dir))
    tensor = torch.zeros(10)
    cache["weight"] = tensor

    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 1
    assert len(files) == 1
    with safe_open(offload_dir / files[0], framework="pt", device="cpu") as file:
        read_tensor = file.get_tensor("weight")
        assert_tensor_equal(read_tensor, tensor)

    # modify
    tensor = torch.ones(10)
    cache["weight"] = tensor

    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 1
    assert len(files) == 1
    with safe_open(offload_dir / files[0], framework="pt", device="cpu") as file:
        read_tensor = file.get_tensor("weight")
        assert_tensor_equal(read_tensor, tensor)

    # delete
    del cache["weight"]
    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 0
    assert len(files) == 0
