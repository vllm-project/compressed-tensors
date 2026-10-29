# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import threading
from collections import OrderedDict
from typing import Iterator

from safetensors import safe_open


__all__ = ["disk_load_context"]


# Open safetensors handles held for the duration of `disk_load_context`, keyed by
# (file read from, device). Thread local because handles are not safe to share
# across threads and callers may prefetch on a background thread.
class _OpenFiles(threading.local):
    """Per-thread handle cache and nesting depth for `disk_load_context`."""

    def __init__(self):
        self.cache: OrderedDict | None = None
        self.depth = 0


_open_files = _OpenFiles()

# Bound on concurrently held handles, so a model with many shards cannot exhaust
# the process file descriptor limit. Reads within one subgraph touch very few
# distinct files, so a small cap gives up nothing in practice.
_MAX_OPEN_FILES = 16


@contextlib.contextmanager
def disk_load_context() -> Iterator[None]:
    """
    Hold safetensors file handles open for the duration of the block.

    `safe_open` parses a header describing every tensor in the file, so the cost
    of opening is proportional to the number of tensors the file holds. Opening
    once per tensor therefore makes a group of N reads from one shard O(N**2).
    Reusing the handle makes the same group linear.

    This is opt in. Outside the context, `DiskCache.onload` opens and closes per
    read exactly as before, so nothing changes for callers that do not use it.

    Nesting is allowed; handles are closed when the outermost context exits.

    Example:
        with disk_load_context():
            for subgraph in subgraphs:
                subgraph(batch)
    """
    if _open_files.depth == 0:
        _open_files.cache = OrderedDict()
    _open_files.depth += 1
    try:
        yield
    finally:
        _open_files.depth -= 1
        if _open_files.depth == 0:
            cache, _open_files.cache = _open_files.cache, None
            for handle in cache.values():
                handle.__exit__(None, None, None)


@contextlib.contextmanager
def _opened(
    file_path: str, device: str, file_key: str | None = None
) -> Iterator["safe_open"]:
    """
    Yield a safetensors handle for `file_path`, reusing an open one if the
    caller is inside `disk_load_context`.

    Outside that context this is an ordinary open/close, which keeps the
    uninstrumented path byte for byte what it was.

    :param file_key: the file `file_path` reads from, when the caller knows it.
        Checkpoint symlinks record their shard in the `DiskCache` index, so N
        tensors from one shard share one handle without resolving each symlink.
        Resolving costs several syscalls per read, each releasing the GIL.
        Defaults to `file_path`.
    """
    cache = _open_files.cache
    if cache is None:
        with safe_open(file_path, framework="pt", device=device) as file:
            yield file
        return

    key = (file_key or file_path, device)
    handle = cache.get(key)
    if handle is None:
        handle = safe_open(file_path, framework="pt", device=device)
        handle.__enter__()
        cache[key] = handle
        # Evict oldest first, never the handle just requested.
        while len(cache) > _MAX_OPEN_FILES:
            _, evicted = cache.popitem(last=False)
            evicted.__exit__(None, None, None)
    else:
        cache.move_to_end(key)
    yield handle


def _evict(file_path: str) -> None:
    """
    Close and drop any cached handle onto the file at `file_path`.

    Must be called before that file's contents are rewritten in place. A handle
    held inside `disk_load_context` otherwise keeps reading the old contents.
    Only files `DiskCache` wrote itself are rewritten in place, and those are
    keyed by their own path.

    Only affects the calling thread's cache, matching `disk_load_context`,
    which is thread local.
    """
    cache = _open_files.cache
    if not cache:
        return
    for key in [key for key in cache if key[0] == file_path]:
        cache.pop(key).__exit__(None, None, None)
