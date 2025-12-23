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
import socket
import subprocess
import sys
import tempfile
import textwrap
from functools import wraps
from typing import Any, Callable


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _in_torchrun_worker(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    # torchrun sets at least RANK/WORLD_SIZE (and usually LOCAL_RANK)
    return "RANK" in env and "WORLD_SIZE" in env


def torchrun(
    *, target: str, world_size: int = 1
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Pytest decorator that runs `target` under torchrun with `world_size` processes.

    - Parent process: spawns torchrun subprocess and fails the test if subprocess fails.
    - Worker process (torchrun): executes the decorated function body.

    `target` format: "some.module.path:test_fn"
    """
    if ":" not in target:
        raise ValueError(
            f"torchrun(target=...) must be 'module.path:function', got: {target!r}"
        )
    mod_name, fn_name = target.split(":", 1)
    mod_name, fn_name = mod_name.strip(), fn_name.strip()
    if not mod_name or not fn_name:
        raise ValueError(f"Invalid torchrun target: {target!r}")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If we're already inside a torchrun worker, just run the test body.
            if _in_torchrun_worker():
                rank = int(os.environ["RANK"])

                import torch

                if torch.cuda.is_available():
                    torch.cuda.set_device(rank)

                return fn(*args, **kwargs)

            # Parent process: launch torchrun and run only `target`.
            port = _free_port()

            runner_code = textwrap.dedent(
                f"""
                import os, sys, importlib, traceback
                import torch
                import torch.distributed as dist

                cwd = os.getcwd()
                if cwd not in sys.path:
                    sys.path.insert(0, cwd)

                mod = importlib.import_module({mod_name!r})
                fn = getattr(mod, {fn_name!r})

                dist.init_process_group()
                torch.cuda.set_device(dist.get_rank())

                rank = int(os.environ.get("RANK", "0"))
                world_size = int(os.environ.get("WORLD_SIZE", "1"))

                try:
                    fn()
                    dist.barrier()
                except SystemExit as e:
                    raise
                except Exception:
                    traceback.print_exc()
                    raise
                """
            ).lstrip()

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                    f.write(runner_code)
                    tmp_path = f.name

                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",  # works everywhere torchrun works
                    "--nproc_per_node",
                    str(world_size),
                    "--master_addr",
                    "127.0.0.1",
                    "--master_port",
                    str(port),
                    tmp_path,
                ]

                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ.copy(),
                )

                if proc.returncode != 0:
                    raise RuntimeError(
                        "torchrun subprocess failed.\n"
                        f"Command: {' '.join(cmd)}\n\n"
                        f"--- stdout ---\n{proc.stdout}\n"
                        f"--- stderr ---\n{proc.stderr}\n"
                    )

                return None
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        return wrapper

    return decorator
