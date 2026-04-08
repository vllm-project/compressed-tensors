# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.distributed import (
    as_broadcastable,
    init_dist,
    is_distributed,
    is_source_process,
)
from compressed_tensors.utils.helpers import deprecated


__all__ = ["is_distributed", "is_rank0", "init_dist", "as_broadcastable"]


@deprecated("compressed_tensors.distributed.utils.py::is_source_process")
def is_rank0() -> bool:
    return is_source_process()
