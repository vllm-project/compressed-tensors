# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded asynchronous writer for disk offloads and checkpoint shards."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from loguru import logger
from safetensors.torch import load_file, save_file


@dataclass
class _WriteRequest:
    path: str
    tensors: dict
    future: Future
    metadata: dict[str, str] | None = None


_STOP = object()


def _size_in_bytes(size: int | str) -> int:
    if isinstance(size, int):
        return size
    match = re.fullmatch(r"\s*(\d+)\s*(B|KB|MB|GB|TB)?\s*", size, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid max shard size: {size!r}")
    amount = int(match.group(1))
    unit = (match.group(2) or "B").upper()
    return amount * (1024 ** ("B", "KB", "MB", "GB", "TB").index(unit))


_ShardJob = tuple[Path, dict[str, torch.Tensor], threading.Event]


class AsyncWriter:
    """Write disk-offload snapshots and optional checkpoint shards asynchronously.

    Callers use :meth:`batch` around a subgraph offload. The producer first
    snapshots all of that subgraph's tensors to CPU, then commits the batch so
    the writer can serialize it while later subgraphs run. At most one queued
    batch is retained in addition to the batch currently being written.

    With ``streaming_save=True``, persistent model tensors are assigned to checkpoint
    shards and the writer publishes their index. Temporary or non-persistent tensors
    continue to use disk-offload files through this same worker.
    """

    INDEX_NAME = "model.safetensors.index.json"
    SHARD_PREFIX = "model-streaming-"

    def __init__(
        self,
        max_queued_batches: int = 1,
        defer_errors: bool | None = None,
        *,
        output_dir: str | os.PathLike | None = None,
        streaming_save: bool = False,
        max_shard_size: int | str = "20GB",
    ):
        """Create a bounded writer, optionally streaming checkpoint shards.

        ``defer_errors`` keeps worker failures from aborting a later submission;
        callers can then report them at a distributed flush point on every rank.
        """
        if max_queued_batches < 1:
            raise ValueError("max_queued_batches must be at least 1")
        if streaming_save and output_dir is None:
            raise ValueError("Streaming checkpoint saving requires output_dir")
        shard_size = _size_in_bytes(max_shard_size)
        if shard_size <= 0:
            raise ValueError("max shard size must be positive")

        distributed = dist.is_available() and dist.is_initialized()
        if defer_errors is None:
            defer_errors = streaming_save and distributed and dist.get_world_size() > 1

        self._defer_errors = bool(defer_errors)
        self.streaming_save = streaming_save
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if streaming_save and output_dir is not None
            else None
        )
        self.max_shard_size = shard_size
        from compressed_tensors.distributed import is_source_process

        self.is_source = is_source_process()
        self.run_id = uuid.uuid4().hex[:12] if streaming_save else None
        if streaming_save and self.is_source:
            assert self.output_dir is not None
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=max_queued_batches)
        self._batch_slots = threading.BoundedSemaphore(max_queued_batches)
        self._condition = threading.Condition()
        self._active_batch: list[_WriteRequest] | None = None
        self._flushing = False
        self._pending: dict[str, Future] = {}
        self._errors: dict[str, BaseException] = {}
        self._closed = False
        self._checkpoint_lock = threading.RLock()
        self._weight_map: dict[str, str] = {}
        self._tensor_sizes: dict[str, tuple[int, int]] = {}
        self._shard_submissions: dict[str, threading.Event] = {}
        self._submission_errors: dict[str, BaseException] = {}
        self._current_path: Path | None = None
        self._current_tensors: dict[str, torch.Tensor] = {}
        self._current_size = 0
        self._shard_number = 0
        self._in_batch = False
        self._published = False
        self._thread = threading.Thread(
            target=self._run, name="compressed-tensors-async-writer", daemon=True
        )
        self._thread.start()

    @contextlib.contextmanager
    def batch(self) -> Iterator[AsyncWriter]:
        """Reserve bounded queue capacity for one subgraph's writes."""
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
            if self.streaming_save and self.is_source:
                with self._checkpoint_lock:
                    self._in_batch = True

        try:
            yield self
        except BaseException as error:
            if self.streaming_save and self.is_source:
                with self._checkpoint_lock:
                    self._in_batch = False
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
            if self.streaming_save and self.is_source:
                with self._checkpoint_lock:
                    self._in_batch = False
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
        """Wait for a cache-file write or checkpoint shard to finish."""
        path = os.fspath(path)
        checkpoint_path = self.is_checkpoint_path(path)
        if checkpoint_path:
            shard_path = Path(path)
            if not self.is_source:
                self._wait_for_shared_file(shard_path)
                return

            with self._checkpoint_lock:
                job = None
                if self._current_path == shard_path:
                    job = self._take_shard_locked()
                event = self._shard_submissions.get(str(shard_path))
            if job is not None:
                self._submit_shard(job)
                event = job[2]
            if event is not None:
                event.wait()
            with self._checkpoint_lock:
                submission_error = self._submission_errors.get(str(shard_path))
            if submission_error is not None:
                raise RuntimeError(
                    f"Failed to submit checkpoint shard {shard_path}"
                ) from submission_error

        with self._condition:
            error = self._errors.get(path)
            future = self._pending.get(path)

        if error is not None:
            raise RuntimeError(f"Asynchronous disk write failed for {path}") from error
        if future is not None:
            future.result()
        if checkpoint_path and not Path(path).exists():
            raise FileNotFoundError(f"No checkpoint shard was assigned at {path}")

    @staticmethod
    def _wait_for_shared_file(path: Path, timeout: float = 600.0) -> None:
        deadline = time.monotonic() + timeout
        error_path = path.with_suffix(path.suffix + ".error")
        while not path.exists():
            if error_path.exists():
                raise RuntimeError(
                    f"Source rank failed to write checkpoint shard {path}: "
                    f"{error_path.read_text(encoding='utf-8')}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for checkpoint shard {path}")
            time.sleep(0.05)

    def add_tensor(self, name: str, tensor: torch.Tensor) -> tuple[str, str]:
        """Snapshot a model tensor and return its assigned shard path and key."""
        if not self.streaming_save:
            raise RuntimeError("Checkpoint tensors require streaming_save=True")
        assert self.output_dir is not None
        if not self.is_source:
            raise RuntimeError("Only the source rank can submit checkpoint tensors")
        if tensor.device.type == "meta":
            raise ValueError(f"Cannot stream meta tensor {name!r}")

        assert self.run_id is not None
        jobs: list[_ShardJob | None] = []
        with self._checkpoint_lock:
            if not self._in_batch:
                raise RuntimeError("Checkpoint tensors must be submitted in a batch")
            if name in self._weight_map:
                raise ValueError(f"Duplicate streamed model tensor: {name}")

            tensor_bytes = tensor.numel() * tensor.element_size()
            if (
                self._current_tensors
                and self._current_size + tensor_bytes > self.max_shard_size
            ):
                jobs.append(self._take_shard_locked())
            if self._current_path is None:
                self._shard_number += 1
                self._current_path = self.output_dir / (
                    f"{self.SHARD_PREFIX}{self.run_id}-"
                    f"{self._shard_number:05d}.safetensors"
                )

            # Snapshot on the producer so CT can replace the parameter with its
            # meta handle while the worker serializes the stable CPU tensor.
            snapshot = tensor.detach().contiguous().to(device="cpu", copy=True)
            self._current_tensors[name] = snapshot
            self._current_size += tensor_bytes
            self._weight_map[name] = self._current_path.name
            self._tensor_sizes[name] = (tensor.numel(), tensor_bytes)
            assigned_path = self._current_path

            if self._current_size >= self.max_shard_size:
                jobs.append(self._take_shard_locked())

        for job in jobs:
            self._submit_shard(job)
        return str(assigned_path), name

    def _take_shard_locked(self) -> _ShardJob | None:
        if not self._current_tensors:
            return None
        assert self._current_path is not None
        path = self._current_path
        tensors = self._current_tensors
        event = threading.Event()
        self._shard_submissions[str(path)] = event
        self._current_path = None
        self._current_tensors = {}
        self._current_size = 0
        return path, tensors, event

    def _submit_shard(self, job: _ShardJob | None) -> None:
        if job is None:
            return
        path, tensors, event = job
        try:
            self.submit_file(path, tensors, metadata={"format": "pt"})
        except BaseException as error:
            with self._checkpoint_lock:
                self._submission_errors[str(path)] = error
            event.set()
            raise
        else:
            event.set()

    def is_checkpoint_path(self, path: str | os.PathLike) -> bool:
        if not self.streaming_save or self.output_dir is None:
            return False
        candidate = Path(path)
        return (
            candidate.parent.resolve() == self.output_dir
            and candidate.name.startswith(self.SHARD_PREFIX)
        )

    def update_tensor(self, path: str | os.PathLike, name: str, tensor: torch.Tensor):
        """Atomically replace a tensor in a streamed shard after finalization."""
        if not self.is_checkpoint_path(path):
            raise ValueError(f"Not a streamed checkpoint shard: {path}")
        if not self.is_source:
            return
        shard_path = Path(path)
        self.wait(shard_path)
        with self._checkpoint_lock:
            tensors = load_file(str(shard_path), device="cpu")
            if name not in tensors:
                raise KeyError(f"Tensor {name!r} is missing from {shard_path}")
            updated = tensor.detach().contiguous().to(device="cpu", copy=True)
            if tensors[name].shape != updated.shape:
                raise ValueError(
                    f"Streaming checkpoint update changed the shape of {name}: "
                    f"{tuple(tensors[name].shape)} to {tuple(updated.shape)}"
                )
            old = tensors[name]
            tensors[name] = updated
            temp_path = shard_path.with_name(
                f".{shard_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                save_file(tensors, str(temp_path), metadata={"format": "pt"})
                os.replace(temp_path, shard_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            self._tensor_sizes[name] = (
                updated.numel(),
                updated.numel() * updated.element_size(),
            )
            del old, tensors

    def wait_all(self) -> None:
        """Flush the partial checkpoint shard and drain all queued writes."""
        if not self.streaming_save:
            self.flush()
            return
        if self.is_source:
            with self._checkpoint_lock:
                job = self._take_shard_locked()
                events = list(self._shard_submissions.values())
            if job is not None:
                self._submit_shard(job)
            for event in events:
                event.wait()
            with self._checkpoint_lock:
                submission_errors = list(self._submission_errors.items())
            if submission_errors:
                path, error = submission_errors[0]
                raise RuntimeError(
                    f"Failed to submit checkpoint shard {path}"
                ) from error
        self.flush()

    def validate_coverage(self, expected_tensor_names: set[str]) -> None:
        """Check that streamed shards contain exactly the model's saved tensors."""
        if not self.streaming_save:
            raise RuntimeError("Tensor coverage requires streaming_save=True")
        if not self.is_source:
            return
        actual = set(self._weight_map)
        missing = expected_tensor_names - actual
        unexpected = actual - expected_tensor_names
        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    f"missing {len(missing)} tensors, e.g. {sorted(missing)[:5]}"
                )
            if unexpected:
                details.append(
                    f"found {len(unexpected)} unexpected tensors, e.g. "
                    f"{sorted(unexpected)[:5]}"
                )
            raise ValueError(
                "Streaming checkpoint tensors do not cover the model state: "
                + "; ".join(details)
                + ". Ensure every saved tensor belongs to a traced subgraph."
            )

    def publish(self) -> None:
        """Write the checkpoint index after all shard files have completed."""
        if not self.streaming_save:
            raise RuntimeError("Publishing requires streaming_save=True")
        self.wait_all()
        if self.is_source:
            assert self.output_dir is not None
            if not self._weight_map:
                raise ValueError("Streaming checkpoint contains no tensors")

            total_size = sum(size for _, size in self._tensor_sizes.values())
            total_parameters = sum(count for count, _ in self._tensor_sizes.values())
            index = {
                "metadata": {
                    "total_parameters": total_parameters,
                    "total_size": total_size,
                },
                "weight_map": self._weight_map,
            }
            index_path = self.output_dir / self.INDEX_NAME
            temp_path = index_path.with_name(
                f".{index_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temp_path.write_text(
                    json.dumps(index, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temp_path, index_path)
                self._published = True
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            keep = set(self._weight_map.values())
            for old_shard in self.output_dir.glob(
                f"{self.SHARD_PREFIX}*.safetensors"
            ):
                if old_shard.name not in keep:
                    old_shard.unlink(missing_ok=True)
            for old_shard in self.output_dir.glob("model-?????-of-?????.safetensors"):
                old_shard.unlink(missing_ok=True)
            (self.output_dir / "model_mtp.safetensors").unlink(missing_ok=True)
            (self.output_dir / "model_backbone.safetensors").unlink(missing_ok=True)
            (self.output_dir / "model.safetensors").unlink(missing_ok=True)
            logger.info(
                "Streamed {} tensors into {} checkpoint shards",
                len(self._weight_map),
                len(keep),
            )

        self.close()

    def abort(self) -> None:
        """Stop the worker and remove checkpoint shards from an incomplete run."""
        try:
            self.wait_all()
        except Exception:
            pass
        try:
            self.close()
        except Exception:
            pass
        if not self.streaming_save or not self.is_source or self._published:
            return
        assert self.output_dir is not None and self.run_id is not None
        pattern = f"{self.SHARD_PREFIX}{self.run_id}-*.safetensors*"
        for path in self.output_dir.glob(pattern):
            path.unlink(missing_ok=True)

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

    def __enter__(self) -> AsyncWriter:
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
