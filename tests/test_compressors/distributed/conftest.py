# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch.distributed as dist
from compressed_tensors.offload.dist_utils import is_distributed


@pytest.fixture
def offload_folder(tmp_path) -> str:
    offload_path = tmp_path / "offload_dir"

    if not is_distributed() or dist.get_rank() == 0:
        os.makedirs(offload_path, exist_ok=True)

    if is_distributed():
        broadcast_object = [str(offload_path)]
        dist.broadcast_object_list(broadcast_object, src=0)
        offload_path = broadcast_object[0]

    return str(offload_path)
