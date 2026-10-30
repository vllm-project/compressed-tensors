# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental asynchronous host/device transfers for sequential offloading."""

from __future__ import annotations

import contextlib
import contextvars
import time
from collections import defaultdict
from collections.abc import Iterator

import torch
from loguru import logger


_active_transfer: contextvars.ContextVar[AsyncTransfer | None] = (
    contextvars.ContextVar("compressed_tensors_async_transfer", default=None)
)


def _device_from_to_args(args, kwargs) -> torch.device | None:
    device = kwargs.get("device")
    if device is None and args:
        candidate = args[0]
        if isinstance(candidate, (str, torch.device, int)):
            device = candidate
    if device is None:
        return None
    try:
        result = torch.device(device)
    except (TypeError, RuntimeError):
        return None
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def current_async_transfer() -> AsyncTransfer | None:
    """Return the transfer manager active in this thread, if any."""
    return _active_transfer.get()


class AsyncTransfer:
    """Queue H2D/D2H copies and profile sequential subgraph transfer phases.

    CPU sources are pinned on demand, and D2H copies target pinned host buffers,
    so the experimental path does not depend on the cache's ``pin_memory`` option.
    Host buffers are retained until the next subgraph's onload waits for the prior
    offload event. All copies use the current stream for the tensor's CUDA device.
    """

    def __init__(self, device: torch.device | str, enabled: bool = True):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.enabled = bool(
            enabled and self.device.type == "cuda" and torch.cuda.is_available()
        )
        self._host_references: list[torch.Tensor] = []
        self._pending_onload_end: torch.cuda.Event | None = None
        self._pending_offload: tuple[
            str, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event
        ] | None = None
        self._ready_event: torch.cuda.Event | None = None
        self._profiles: dict[str, dict[str, object]] = defaultdict(dict)
        self._pin_fallbacks = 0
        if self.enabled:
            logger.info(
                "Experimental async onload/offload enabled on {}. CUDA event times "
                "measure stream elapsed time; host phase wall times can include "
                "implicit synchronization.",
                self.device,
            )

    @contextlib.contextmanager
    def _activate(self) -> Iterator[None]:
        token = _active_transfer.set(self)
        try:
            yield
        finally:
            _active_transfer.reset(token)

    @staticmethod
    def _record(event: torch.cuda.Event, device: torch.device) -> None:
        with torch.cuda.device(device):
            event.record(torch.cuda.current_stream(device))

    def move_tensor(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Move one tensor asynchronously when it crosses CPU/CUDA memory."""
        device = _device_from_to_args(args, kwargs)
        if not self.enabled or device is None:
            return tensor.to(*args, **kwargs)

        if tensor.device.type == "cpu" and device.type == "cuda":
            if device != self.device:
                return tensor.to(*args, **kwargs)
            source = tensor
            if not source.is_pinned():
                try:
                    source = source.pin_memory()
                except RuntimeError as error:
                    self._log_pin_fallback(error)
                    return tensor.to(*args, **kwargs)
            self._host_references.append(source)
            async_kwargs = dict(kwargs)
            async_kwargs["non_blocking"] = True
            return source.to(*args, **async_kwargs)

        if tensor.device.type == "cuda" and device.type == "cpu":
            if tensor.device != self.device:
                return tensor.to(*args, **kwargs)
            return self.to_cpu(
                tensor,
                copy=kwargs.get("copy", False),
                contiguous=False,
            )

        return tensor.to(*args, **kwargs)

    def to_cpu(
        self,
        tensor: torch.Tensor,
        *,
        copy: bool = True,
        contiguous: bool = False,
    ) -> torch.Tensor:
        """Copy a tensor to CPU, using a pinned destination for asynchronous D2H."""
        if (
            not self.enabled
            or tensor.device.type != "cuda"
            or tensor.device != self.device
        ):
            result = tensor.detach().to(device="cpu", copy=copy)
            return result.contiguous() if contiguous else result

        source = tensor.detach()
        try:
            with torch.cuda.device(source.device):
                if contiguous:
                    source = source.contiguous()
                destination = torch.empty_like(
                    source,
                    device="cpu",
                    pin_memory=True,
                    memory_format=torch.preserve_format,
                )
                destination.copy_(source, non_blocking=True)
        except RuntimeError as error:
            self._log_pin_fallback(error)
            result = tensor.detach().contiguous().to(device="cpu", copy=True)
            return result

        self._host_references.extend((source, destination))
        return destination

    def _log_pin_fallback(self, error: RuntimeError) -> None:
        self._pin_fallbacks += 1
        if self._pin_fallbacks == 1:
            logger.warning(
                "Pinned host allocation failed; falling back to a blocking copy "
                "for this and any later affected tensors: {}",
                error,
            )

    @contextlib.contextmanager
    def onloading(self, label: str) -> Iterator[None]:
        """Record H2D copies issued while onloading one subgraph."""
        if not self.enabled:
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self._record(start, self.device)
        started = time.perf_counter()
        error = None
        try:
            with self._activate():
                try:
                    yield
                except BaseException as caught:
                    error = caught
                    raise
                finally:
                    self._record(end, self.device)
        finally:
            if error is not None:
                end.synchronize()
                self._host_references.clear()
        self._profiles[label]["host_onload_submit_ms"] = (
            time.perf_counter() - started
        ) * 1000
        self._profiles[label]["_onload_start"] = start
        self._profiles[label]["_onload_end"] = end
        self._pending_onload_end = end

    @contextlib.contextmanager
    def offloading(self, label: str, writer=None) -> Iterator[None]:
        """Record D2H copies and expose their completion event to ``AsyncWriter``.

        The writer batch is committed only after the event has been recorded, so
        its worker can safely synchronize the event before reading host buffers.
        """
        if not self.enabled:
            if writer is None:
                yield
            else:
                with writer.batch():
                    yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event()
        started = time.perf_counter()
        self._ready_event = ready
        writer_context = (
            writer.batch(ready_event=ready, label=label)
            if writer is not None
            else contextlib.nullcontext()
        )
        error = None
        try:
            with writer_context:
                self._record(start, self.device)
                with self._activate():
                    try:
                        yield
                    except BaseException as caught:
                        error = caught
                        raise
                    finally:
                        self._record(end, self.device)
                        self._record(ready, self.device)
                        self._profiles[label]["host_offload_submit_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                        self._profiles[label]["_offload_start"] = start
                        self._profiles[label]["_offload_end"] = end
                        self._pending_offload = (label, ready, start, end)
        finally:
            self._ready_event = None
            if error is not None:
                # The failed batch will not be written. Keep in-flight host and
                # device buffers safe before stack unwinding releases references.
                ready.synchronize()
                self._host_references.clear()

    @property
    def ready_event(self) -> torch.cuda.Event | None:
        """Completion event for the offload batch currently being submitted."""
        return self._ready_event

    def record_host_metrics(self, label: str, **metrics: float) -> None:
        """Add wall-clock measurements collected by the sequential pipeline."""
        if not self.enabled:
            return
        self._profiles[label].update(metrics)

    @contextlib.contextmanager
    def cuda_phase(self, label: str, phase: str) -> Iterator[None]:
        """Measure elapsed time on the CUDA stream around a pipeline phase."""
        if not self.enabled:
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self._record(start, self.device)
        try:
            yield
        finally:
            self._record(end, self.device)
            self._profiles[label][f"__cuda_{phase}_events"] = (start, end)

    def wait_previous_offload(self) -> None:
        """Block before another onload, bounding live GPU subgraph weights."""
        if not self.enabled:
            return
        if self._pending_offload is None:
            if self._pending_onload_end is not None:
                self._pending_onload_end.synchronize()
                self._pending_onload_end = None
                self._host_references.clear()
            return

        label, ready, start, end = self._pending_offload
        wait_started = time.perf_counter()
        ready.synchronize()
        self._profiles[label]["blocking_wait_previous_subgraph_ms"] = (
            time.perf_counter() - wait_started
        ) * 1000
        onload_start = self._profiles[label].pop("_onload_start", None)
        onload_end = self._profiles[label].pop("_onload_end", None)
        if onload_start is not None and onload_end is not None:
            self._profiles[label]["cuda_h2d_stream_elapsed_ms"] = (
                onload_start.elapsed_time(onload_end)
            )
        self._profiles[label]["cuda_d2h_stream_elapsed_ms"] = start.elapsed_time(end)
        metrics = self._profiles.pop(label)
        for name, events in list(metrics.items()):
            if name.startswith("__cuda_") and name.endswith("_events"):
                phase = name.removeprefix("__cuda_").removesuffix("_events")
                phase_start, phase_end = events
                metrics[f"cuda_{phase}_stream_elapsed_ms"] = (
                    phase_start.elapsed_time(phase_end)
                )
                metrics.pop(name)
        metrics.pop("_offload_start", None)
        metrics.pop("_offload_end", None)
        logger.info(
            "Sequential transfer profile [{}]: {}",
            label,
            ", ".join(
                f"{name}={value:.2f} ms"
                for name, value in metrics.items()
                if isinstance(value, (int, float))
            ),
        )
        self._pending_offload = None
        self._pending_onload_end = None
        self._host_references.clear()
