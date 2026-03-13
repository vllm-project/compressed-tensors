# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
from compressed_tensors.offload import OffloadCache


@pytest.fixture()
def offload_cache(offload_device, onload_device, tmp_path):
    if offload_device == "disk":
        offload_dir = str(tmp_path / "offload_dir")
        os.makedirs(offload_dir)
        return OffloadCache.cls_from_device(offload_device)(
            onload_device, offload_dir=offload_dir
        )
    else:
        return OffloadCache.cls_from_device(offload_device)(onload_device)
