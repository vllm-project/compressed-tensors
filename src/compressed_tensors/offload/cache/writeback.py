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

import torch.distributed as dist
from safetensors.torch import save_file


@dataclass
class _WriteRequest:
    path: str
    tensors: dict
    future: Future
    metadata: dict[str, str] | None = None


_STOP = object()


class AsyncDiskWriteback:
    """Serialize subgraph CPU snapshots on one bounded background thread.

    Callers use :meth:`batch` around a subgraph offload. The producer first
    snapshots all of that subgraph's tensors to CPU, then commits the batch so
    the writer can serialize it while later subgraphs run. At most one queued
    batch is retained in addition to the batch currently being written.

    A single batch is the entire subgraph to be written.
    """

    def __init__(self, max_queued_batches: int = 1, defer_errors: bool = False):
        """Create the bounded writer.

        ``defer_errors`` keeps worker failures from aborting a later submission;
        callers can then report them at a distributed flush point on every rank.
        """
        if max_queued_batches < 1:
            raise ValueError("max_queued_batches must be at least 1")

        self._defer_errors = defer_errors
        self._queue: queue.Queue = queue.Queue(maxsize=max_queued_batches)
        self._batch_slots = threading.BoundedSemaphore(max_queued_batches)
        self._condition = threading.Condition()
        self._active_batch: list[_WriteRequest] | None = None
        self._flushing = False
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
            while self._flushing:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("Cannot submit to a closed disk writeback worker")
            if self._active_batch is not None:
                raise RuntimeError("Disk writeback batches cannot be nested")

        self._batch_slots.acquire()
        with self._condition:
            while self._flushing:
                self._condition.wait()
            if self._closed:
                self._batch_slots.release()
                raise RuntimeError("Cannot submit to a closed disk writeback worker")
            if self._active_batch is not None:
                self._batch_slots.release()
                raise RuntimeError("Disk writeback batches cannot be nested")
            try:
                if not self._defer_errors:
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
                failed_requests = list(batch)
                for request in failed_requests:
                    self._pending.pop(request.path, None)
                    self._errors[request.path] = error
                    request.future.set_exception(error)
                self._active_batch = None
                self._condition.notify_all()
            self._batch_slots.release()
            for request in failed_requests:
                self._write_error_marker(request.path, error)
            raise
        else:
            with self._condition:
                self._active_batch = None
                if batch:
                    # The semaphore reserves a queue slot for this batch. Enqueue
                    # before waking flush callers so they cannot miss this batch.
                    self._queue.put(batch)
                else:
                    self._batch_slots.release()
                self._condition.notify_all()

    def submit(
        self,
        path: str,
        tensors: dict,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Add a stable, contiguous CPU mapping to the active subgraph batch."""
        path = os.fspath(path)
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot submit to a closed disk writeback worker")
            if self._active_batch is None:
                raise RuntimeError("Disk tensor snapshots must be submitted in a batch")
            self._submit_locked(path, tensors, metadata)

    def submit_file(
        self,
        path: str | os.PathLike,
        tensors: dict,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Submit a file write to the active subgraph or as a bounded singleton.

        Checkpoint shards can be flushed by a later staging thread after the
        subgraph's offload batch has ended. In that case, enqueue the write through
        the same bounded worker used for per-tensor disk-cache files.
        """
        path = os.fspath(path)
        with self._condition:
            while self._flushing and self._active_batch is None:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("Cannot submit to a closed disk writeback worker")
            if self._active_batch is not None:
                self._submit_locked(path, tensors, metadata)
                return

        # Reserve queue capacity without holding the condition. A batch may begin
        # while this thread waits, so check for one again after acquiring the slot.
        self._batch_slots.acquire()
        with self._condition:
            while self._flushing and self._active_batch is None:
                self._condition.wait()
            if self._closed:
                self._batch_slots.release()
                raise RuntimeError("Cannot submit to a closed disk writeback worker")
            if self._active_batch is not None:
                try:
                    self._submit_locked(path, tensors, metadata)
                finally:
                    self._batch_slots.release()
                return
            try:
                if not self._defer_errors:
                    self._raise_first_error_locked()
                request = self._new_request_locked(path, tensors, metadata)
            except BaseException:
                self._batch_slots.release()
                raise
            # The reserved semaphore slot guarantees this insertion can proceed
            # without waiting for the worker while holding the condition.
            self._queue.put([request])
            self._condition.notify_all()

    def _submit_locked(
        self, path: str, tensors: dict, metadata: dict[str, str] | None
    ) -> None:
        if not self._defer_errors:
            self._raise_first_error_locked()
        request = self._new_request_locked(path, tensors, metadata)
        assert self._active_batch is not None
        self._active_batch.append(request)

    def _new_request_locked(
        self, path: str, tensors: dict, metadata: dict[str, str] | None
    ) -> _WriteRequest:
        if path in self._pending:
            raise RuntimeError(f"A disk write is already pending for {path}")
        future = Future()
        request = _WriteRequest(path, tensors, future, metadata)
        self._pending[path] = future
        return request

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
        with self._condition:
            while self._flushing:
                self._condition.wait()
            self._flushing = True
            while self._active_batch is not None:
                self._condition.wait()
        try:
            self._queue.join()
            with self._condition:
                self._raise_first_error_locked()
        finally:
            with self._condition:
                self._flushing = False
                self._condition.notify_all()

    def flush_distributed(self, group) -> None:
        """Flush local writes and report any failure to every rank in ``group``.

        The group should be independent from the process group used for model
        collectives when this is called from a staging thread.
        """
        local_error = None
        try:
            self.flush()
        except BaseException as error:
            local_error = error

        local_message = (
            None
            if local_error is None
            else f"{type(local_error).__name__}: {local_error}"
        )
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, local_message, group=group)

        failures = [
            f"rank {rank}: {message}"
            for rank, message in enumerate(errors)
            if message is not None
        ]
        if failures:
            raise RuntimeError(
                "Asynchronous disk writeback failed on " + "; ".join(failures)
            ) from local_error

    def close(self) -> None:
        """Drain queued writes and stop the worker."""
        with self._condition:
            while self._flushing:
                self._condition.wait()
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
                        self._write_atomically(
                            request.path, request.tensors, request.metadata
                        )
                    except BaseException as error:
                        request.future.set_exception(error)
                        with self._condition:
                            self._errors[request.path] = error
                        self._write_error_marker(request.path, error)
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
    def _write_error_marker(path: str, error: BaseException) -> None:
        marker = Path(path).with_suffix(Path(path).suffix + ".error")
        try:
            marker.write_text(repr(error), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _write_atomically(
        path: str,
        tensors: dict,
        metadata: dict[str, str] | None = None,
    ) -> None:
        directory = os.path.dirname(path)
        prefix = f".{Path(path).name}."
        fd, temporary_path = tempfile.mkstemp(
            dir=directory, prefix=prefix, suffix=".tmp"
        )
        os.close(fd)

        try:
            save_file(tensors, temporary_path, metadata=metadata)
            os.replace(temporary_path, path)
            error_path = Path(path).with_suffix(Path(path).suffix + ".error")
            error_path.unlink(missing_ok=True)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
