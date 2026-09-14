# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.offload.utils import index_from_view


def test_index_from_view():
    a = torch.rand(10, 10, 10)
    b = a[0, ::2, :5]
    base, index = index_from_view(b)

    assert torch.equal(base, a)
    assert torch.equal(base[index], b)
