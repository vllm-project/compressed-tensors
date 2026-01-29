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
import subprocess
import sys
from functools import wraps
from types import FunctionType
from typing import Any, Callable

import torch
import torch.distributed as dist


def torchrun(world_size: int = 1) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Test a function within parallel `torchrun` subprocesses, each running with `pytest`
    ```
    # (main) -> (torchrun) -
    #              \\-- (rank 0)
    #              \\-- (rank 1)
    ```

    :param world_size: number of ranks to spawn
    """

    def decorator(func: FunctionType):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # We're running in a torchrun subprocess:
            # init distributed and run test func
            if "TORCHELASTIC_RUN_ID" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])
                rank = int(os.environ["RANK"])

                torch.cuda.set_device(local_rank)
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    rank=rank,
                    world_size=world_size,
                )
                dist.barrier()

                return func(*args, **kwargs)

            # First time calling in the main process:
            # trigger torchrun with this function as the pytest target
            else:
                file_path = sys.modules.get(func.__module__).__file__
                func_name = func.__name__

                cmd = (
                    f"{sys.executable} "
                    f"-m torch.distributed.run --nproc_per_node {world_size} "
                    "--log-dir /tmp/torchrun-logs --tee 3 --role torchrun "
                    f"-m pytest {file_path}::{func_name} -sx"
                )

                proc = subprocess.run(cmd.split(" "))
                assert proc.returncode == 0

        return wrapper

    return decorator
