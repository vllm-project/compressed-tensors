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


# Raw file readers held for the duration of `disk_load_context`, keyed by
# resolved path. Thread local because readers are not safe to share across
# threads and callers may prefetch on a background thread.
class _OpenFiles(threading.local):
    """Per-thread reader cache and nesting depth for `disk_load_context`."""

    def __init__(self):
        self.raw_cache: OrderedDict | None = None
        self.depth = 0


_open_files = _OpenFiles()

# Bound on concurrently held readers, so a model with many shards cannot exhaust
# the process file descriptor limit. Reads within one subgraph touch very few
# distinct files, so a small cap gives up nothing in practice.
_MAX_OPEN_FILES = 16


@contextlib.contextmanager
def disk_load_context() -> Iterator[None]:
    """
    Hold raw file readers open for the duration of the block.

    A reader parses a header describing every tensor in the file, so the cost
    of opening is proportional to the number of tensors the file holds. Opening
    once per tensor therefore makes a group of N reads from one shard O(N**2).
    Reusing the reader makes the same group linear.

    This is opt in. Outside the context, reads open and close per reader
    exactly as before, so nothing changes for callers that do not use it.

    Nesting is allowed; readers are closed when the outermost context exits.

    Example:
        with disk_load_context():
            for subgraph in subgraphs:
                subgraph(batch)
    """
    if _open_files.depth == 0:
        _open_files.raw_cache = OrderedDict()
    _open_files.depth += 1
    try:
        yield
    finally:
        _open_files.depth -= 1
        if _open_files.depth == 0:
            raw_cache, _open_files.raw_cache = _open_files.raw_cache, None
            for file, _, _ in raw_cache.values():
                file.close()


@contextlib.contextmanager
def _opened_raw(file_path: str) -> Iterator[tuple]:
    """
    Yield `(file, header, data_offset)` for reading raw tensor bytes, reusing
    an open reader inside `disk_load_context`.

    The safetensors format fixes the header length to a little-endian uint64
    followed by a JSON header; tensor `data_offsets` in the header are relative
    to the end of the header. Unlike `safe_open`, the returned reader allows
    reading a payload directly into caller-owned memory (used to read into
    page-locked buffers without an intermediate copy).

    Readers are positioned per read; the same reader must not be shared across
    threads. Outside the context, the reader is closed after each use.
    """
    cache = _open_files.raw_cache
    key = _file_key(file_path)
    entry = cache.get(key) if cache is not None else None
    if entry is None:
        file = open(file_path, "rb", buffering=0)
        try:
            header_size = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(header_size))
            data_offset = 8 + header_size
        except BaseException:
            file.close()
            raise
        entry = (file, header, data_offset)
        if cache is not None:
            cache[key] = entry
            while len(cache) > _MAX_OPEN_FILES:
                _, evicted = cache.popitem(last=False)
                evicted[0].close()
    else:
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
    Close and drop any cached reader onto the file at `file_path`.

    Must be called before that file's contents are rewritten in place. A reader
    held inside `disk_load_context` otherwise keeps reading the old contents.

    Only affects the calling thread's cache, matching `disk_load_context`,
    which is thread local.
    """
    raw_cache = _open_files.raw_cache
    if not raw_cache:
        return
    entry = raw_cache.pop(_file_key(file_path), None)
    if entry is not None:
        entry[0].close()
