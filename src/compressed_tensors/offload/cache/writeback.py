# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded background writer for CPU snapshots from disk-backed modules."""

from __future__ import annotations

import contextlib
import os
import queue
import tempfile
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from safetensors.torch import save_file


@dataclass
class _WriteRequest:
    path: str
    tensors: dict
    future: Future


_STOP = object()


class AsyncDiskWriteback:
    """Serialize subgraph CPU snapshots on one bounded background thread.

    Callers use :meth:`batch` around a subgraph offload. The producer first
    snapshots all of that subgraph's tensors to CPU, then commits the batch so
    the writer can serialize it while later subgraphs run. At most one queued
    batch is retained in addition to the batch currently being written.
    """

    def __init__(self, max_queued_batches: int = 1):
        if max_queued_batches < 1:
            raise ValueError("max_queued_batches must be at least 1")

        self._queue: queue.Queue = queue.Queue(maxsize=max_queued_batches)
        self._batch_slots = threading.BoundedSemaphore(max_queued_batches)
        self._condition = threading.Condition()
        self._active_batch: list[_WriteRequest] | None = None
        self._pending: dict[str, Future] = {}
        self._errors: dict[str, BaseException] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="compressed-tensors-disk-writeback", daemon=True
        )
        self._thread.start()

    @contextlib.contextmanager
    def batch(self) -> Iterator[AsyncDiskWriteback]:
        """Reserve bounded queue capacity and collect one subgraph's snapshots."""
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot submit to a closed disk writeback worker")
            if self._active_batch is not None:
                raise RuntimeError("Disk writeback batches cannot be nested")

        self._batch_slots.acquire()
        with self._condition:
            if self._closed:
                self._batch_slots.release()
                raise RuntimeError("Cannot submit to a closed disk writeback worker")
            if self._active_batch is not None:
                self._batch_slots.release()
                raise RuntimeError("Disk writeback batches cannot be nested")
            try:
                self._raise_first_error_locked()
            except BaseException:
                self._batch_slots.release()
                raise
            batch: list[_WriteRequest] = []
            self._active_batch = batch

        try:
            yield self
        except BaseException as error:
            with self._condition:
                for request in batch:
                    self._pending.pop(request.path, None)
                    self._errors[request.path] = error
                    request.future.set_exception(error)
                self._active_batch = None
            self._batch_slots.release()
            raise
        else:
            with self._condition:
                self._active_batch = None
            if batch:
                # The semaphore reserves a queue slot for this batch.
                self._queue.put(batch)
            else:
                self._batch_slots.release()

    def submit(self, path: str, tensors: dict) -> None:
        """Add a stable, contiguous CPU mapping to the active subgraph batch."""
        path = os.fspath(path)
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot submit to a closed disk writeback worker")
            if self._active_batch is None:
                raise RuntimeError("Disk tensor snapshots must be submitted in a batch")
            self._raise_first_error_locked()
            if path in self._pending:
                raise RuntimeError(f"A disk write is already pending for {path}")
            future = Future()
            request = _WriteRequest(path, tensors, future)
            self._pending[path] = future
            self._active_batch.append(request)

    @property
    def is_batch_active(self) -> bool:
        with self._condition:
            return self._active_batch is not None

    def wait(self, path: str | os.PathLike) -> None:
        """Wait for a write to ``path`` if it is still in flight."""
        path = os.fspath(path)
        with self._condition:
            error = self._errors.get(path)
            future = self._pending.get(path)

        if error is not None:
            raise RuntimeError(f"Asynchronous disk write failed for {path}") from error
        if future is not None:
            future.result()

    def flush(self) -> None:
        """Wait for queued writes and propagate the first write error."""
        self._queue.join()
        with self._condition:
            self._raise_first_error_locked()

    def close(self) -> None:
        """Drain queued writes and stop the worker."""
        with self._condition:
            if self._active_batch is not None:
                raise RuntimeError("Cannot close with an uncommitted writeback batch")
            if self._closed:
                should_join = True
            else:
                self._closed = True
                should_join = False

        if not should_join:
            # The sentinel follows all committed batches. Do not hold the
            # condition while waiting for queue space; the worker needs it to
            # publish any write errors.
            self._queue.put(_STOP)
            self._queue.join()
            self._thread.join()

        self.flush()

    def __enter__(self) -> AsyncDiskWriteback:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except BaseException as error:
            if exc_value is None:
                raise
            if hasattr(exc_value, "add_note"):
                exc_value.add_note(
                    f"Disk writeback also failed during shutdown: {error}"
                )

    def _run(self) -> None:
        while True:
            batch = self._queue.get()
            try:
                if batch is _STOP:
                    return

                # Once the worker has taken a batch, its queue slot can be
                # reserved by the producer while this batch is being written.
                self._batch_slots.release()
                for request in batch:
                    try:
                        self._write_atomically(request.path, request.tensors)
                    except BaseException as error:
                        request.future.set_exception(error)
                        with self._condition:
                            self._errors[request.path] = error
                    else:
                        request.future.set_result(None)
                    finally:
                        with self._condition:
                            self._pending.pop(request.path, None)
            finally:
                self._queue.task_done()

    def _raise_first_error_locked(self) -> None:
        error = next(iter(self._errors.items()), None)
        if error is not None:
            path, cause = error
            raise RuntimeError(f"Asynchronous disk write failed for {path}") from cause

    @staticmethod
    def _write_atomically(path: str, tensors: dict) -> None:
        directory = os.path.dirname(path)
        prefix = f".{Path(path).name}."
        fd, temporary_path = tempfile.mkstemp(
            dir=directory, prefix=prefix, suffix=".tmp"
        )
        os.close(fd)

        try:
            save_file(tensors, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
