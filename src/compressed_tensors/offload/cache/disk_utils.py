# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import os
import struct
import threading
from collections import OrderedDict
from typing import Iterator


__all__ = ["disk_load_context"]


# Open file handles held for the duration of `disk_load_context`, keyed by
# resolved path. Thread local because handles are not safe to share across
# threads and callers may prefetch on a background thread.
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
    Hold raw file handles open for the duration of the block.

    Opening a file parses a header describing every tensor in the file, so the
    cost of opening is proportional to the number of tensors the file holds.
    Opening once per tensor therefore makes a group of N reads from one shard
    O(N**2). Reusing the handle makes the same group linear.

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
            for handle, _, _ in cache.values():
                handle.close()


@contextlib.contextmanager
def _opened(file_path: str) -> Iterator[tuple]:
    """
    Yield (handle, header, data_offset), reusing an open reader inside
    `disk_load_context`.

    The reader is an unbuffered byte stream. The format fixes the header length
    to a little-endian uint64; tensor offsets in the JSON header are relative
    to the data section.

    Outside the context, close the reader after each use.
    """
    cache = _open_files.cache
    key = _file_key(file_path)
    entry = cache.get(key) if cache is not None else None
    if entry is None:
        handle = open(file_path, "rb", buffering=0)
        try:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
            data_offset = 8 + header_size
        except BaseException:
            handle.close()
            raise
        entry = (handle, header, data_offset)
        if cache is not None:
            cache[key] = entry
            while len(cache) > _MAX_OPEN_FILES:
                _, evicted = cache.popitem(last=False)
                evicted[0].close()
    elif cache is not None:
        cache.move_to_end(key)
    try:
        yield entry
    finally:
        if cache is None:
            entry[0].close()


def _file_key(file_path: str) -> str:
    """
    Identify the file a path actually reads from.

    `DiskCache.create_checkpoint_symlink` gives every tensor its own symlink,
    named after `id(offloaded)`, pointing into a shared checkpoint shard. Keying
    on the path as given would treat N tensors from one shard as N files, so the
    cache would never hit on exactly the path where grouping matters.
    """
    return os.path.realpath(file_path)


def _evict(file_path: str) -> None:
    """
    Close and drop any cached handle onto the file at `file_path`.

    Must be called before that file's contents are rewritten in place. A handle
    held inside `disk_load_context` otherwise keeps reading the old contents.

    Only affects the calling thread's cache, matching `disk_load_context`,
    which is thread local.
    """
    cache = _open_files.cache
    if not cache:
        return
    target = _file_key(file_path)
    for key in [key for key in cache if key == target]:
        cache.pop(key)[0].close()
