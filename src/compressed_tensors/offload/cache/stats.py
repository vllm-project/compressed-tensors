# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class DevicePairStats:
    """Statistics for a specific device pair (source, destination)"""

    count: int = 0  # Total operations for this device pair
    noop_count: int = 0  # No-op operations for this device pair
    bytes_moved: int = 0  # Actual bytes transferred (excluding no-ops)
    noop_bytes: int = 0  # Bytes that would have moved in no-ops (tensor size)


@dataclass
class OperationStats:
    """Statistics for a single operation type (onload, offload, or update)"""

    count: int = 0
    bytes_moved: int = 0
    noop_count: int = 0
    device_stats: defaultdict[tuple[str, str], DevicePairStats] = field(
        default_factory=lambda: defaultdict(DevicePairStats)
    )

    def record(
        self,
        input_tensor: torch.Tensor | None,
        result_tensor: torch.Tensor | None,
    ):
        """
        Record an operation on tensors

        :param input_tensor: input tensor to the operation
        :param result_tensor: result tensor from the operation
        """
        is_noop = input_tensor is result_tensor

        # Get device information
        src_device = self._get_device_str(input_tensor)
        dst_device = self._get_device_str(result_tensor)
        pair_stats = self.device_stats[(src_device, dst_device)]

        # track counts (both top-level and per-device-pair)
        self.count += 1
        pair_stats.count += 1
        if is_noop:
            self.noop_count += 1
            pair_stats.noop_count += 1

        # track bytes (both top-level and per-device-pair)
        if result_tensor is not None:
            bytes_transferred = result_tensor.element_size() * result_tensor.numel()
            if is_noop:
                pair_stats.noop_bytes += bytes_transferred
            else:
                self.bytes_moved += bytes_transferred
                pair_stats.bytes_moved += bytes_transferred

    @staticmethod
    def _get_device_str(tensor: torch.Tensor | None) -> str:
        """
        Get a string representation of the tensor's device

        :param tensor: tensor to get device from
        :return: device string (e.g., 'cuda:0', 'cpu', 'meta', or 'none')
        """
        if tensor is None:
            return "none"
        return str(tensor.device)


class OffloadStats:
    """
    Global statistics tracker for OffloadCache operations.

    This class should never be instantiated. All methods are class methods and all
    statistics are stored as class variables.

    Tracks the number of onload, offload, and update operations, as well as
    the number of bytes affected and whether operations were no-ops.

    Statistics collection is disabled by default to avoid runtime overhead.
    Use enable() to turn on collection and disable() to turn it off.

    Example usage:
        # Enable statistics collection
        OffloadStats.enable()

        # Statistics are collected automatically when decorators are applied
        stats = OffloadStats.get_stats()
        print(stats)

        # Reset statistics
        OffloadStats.reset()

        # Get formatted summary
        summary = OffloadStats.format_summary()
        print(summary)

        # Disable statistics collection
        OffloadStats.disable()
    """

    # Class-level statistics
    _enabled: bool = False
    onload: OperationStats = OperationStats()
    offload: OperationStats = OperationStats()
    update: OperationStats = OperationStats()

    def __init__(self):
        """Prevent instantiation of this class"""
        raise RuntimeError(
            "OffloadStats should not be instantiated. "
            "Use class methods directly (e.g., OffloadStats.get_stats())"
        )

    @classmethod
    def get_stats(cls) -> dict[str, OperationStats]:
        """
        Get the current statistics

        :return: dictionary with 'onload', 'offload', and 'update' statistics
        """
        return {
            "onload": cls.onload,
            "offload": cls.offload,
            "update": cls.update,
        }

    @classmethod
    def get_device_stats(
        cls,
    ) -> dict[str, dict[tuple[str, str], dict[str, int]]]:
        """
        Get device-specific movement statistics

        :return: nested dictionary with structure:
            {
                'onload': {
                    (source_device, dest_device): {
                        'count': int,
                        'noop_count': int,
                        'bytes_moved': int,
                        'noop_bytes': int
                    }
                },
                'offload': {...},
                'update': {...}
            }
        """
        result = {}
        for op_name, op_stats in [
            ("onload", cls.onload),
            ("offload", cls.offload),
            ("update", cls.update),
        ]:
            result[op_name] = {}
            for device_pair, pair_stats in op_stats.device_stats.items():
                result[op_name][device_pair] = {
                    "count": pair_stats.count,
                    "noop_count": pair_stats.noop_count,
                    "bytes_moved": pair_stats.bytes_moved,
                    "noop_bytes": pair_stats.noop_bytes,
                }
        return result

    @classmethod
    def reset(cls):
        """Reset all statistics to zero"""
        cls.onload = OperationStats()
        cls.offload = OperationStats()
        cls.update = OperationStats()

    @classmethod
    def enable(cls):
        """Enable statistics collection"""
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Disable statistics collection"""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """
        Check if statistics collection is enabled

        :return: True if statistics collection is enabled, False otherwise
        """
        return cls._enabled

    @classmethod
    def format_summary(cls, unit: str = "MB", show_devices: bool = False) -> str:
        """
        Generate a formatted summary of device movement statistics

        :param unit: unit for displaying bytes ('B', 'KB', 'MB', or 'GB')
        :param show_devices: whether to include device-specific breakdown
        :return: formatted summary string
        """
        # Conversion factors
        units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
        if unit not in units:
            raise ValueError(f"Unit must be one of {list(units.keys())}")
        divisor = units[unit]

        # Calculate totals
        total_ops = cls.onload.count + cls.offload.count + cls.update.count
        total_bytes = (
            cls.onload.bytes_moved + cls.offload.bytes_moved + cls.update.bytes_moved
        )
        total_noops = (
            cls.onload.noop_count + cls.offload.noop_count + cls.update.noop_count
        )

        # Format summary
        lines = [
            "OffloadCache Statistics",
            "=" * 50,
            f"{'Operation':<12} {'Count':>8} {'No-ops':>8} {'Data Moved':>12}",
            "-" * 50,
            f"{'Onload':<12} {cls.onload.count:>8} "
            f"{cls.onload.noop_count:>8} "
            f"{cls.onload.bytes_moved / divisor:>10.2f} {unit}",
            f"{'Offload':<12} {cls.offload.count:>8} "
            f"{cls.offload.noop_count:>8} "
            f"{cls.offload.bytes_moved / divisor:>10.2f} {unit}",
            f"{'Update':<12} {cls.update.count:>8} "
            f"{cls.update.noop_count:>8} "
            f"{cls.update.bytes_moved / divisor:>10.2f} {unit}",
            "-" * 50,
            f"{'Total':<12} {total_ops:>8} {total_noops:>8} "
            f"{total_bytes / divisor:>10.2f} {unit}",
            "=" * 50,
        ]

        if show_devices:
            lines.extend(["", cls._format_device_breakdown(unit)])

        return "\n".join(lines)

    @classmethod
    def _format_device_breakdown(cls, unit: str) -> str:
        """
        Generate a formatted breakdown of device movements

        :param unit: unit for displaying bytes
        :return: formatted device breakdown string
        """
        divisor = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]

        lines = [
            "Device Movement Breakdown",
            "=" * 105,
            f"{'Operation':<12} {'Source':>12} {'Dest':>12} {'Count':>8} "
            f"{'No-ops':>8} {'Moved':>12} {'No-op Data':>12}",
            "-" * 105,
        ]

        # Collect all device movements
        for op_name, op_stats in [
            ("Onload", cls.onload),
            ("Offload", cls.offload),
            ("Update", cls.update),
        ]:
            if op_stats.device_stats:
                for (src, dst), pair_stats in sorted(op_stats.device_stats.items()):
                    lines.append(
                        f"{op_name:<12} {src:>12} {dst:>12} {pair_stats.count:>8} "
                        f"{pair_stats.noop_count:>8} "
                        f"{pair_stats.bytes_moved / divisor:>10.2f} {unit} "
                        f"{pair_stats.noop_bytes / divisor:>10.2f} {unit}"
                    )

        lines.append("=" * 105)
        return "\n".join(lines)

    @classmethod
    def track_onload(cls, func: Callable) -> Callable:
        """
        Decorator to track onload operations

        :param func: onload method to decorate
        :return: decorated method
        """

        @functools.wraps(func)
        def wrapper(self, offloaded: torch.Tensor | None) -> torch.Tensor | None:
            result = func(self, offloaded)
            if cls._enabled:
                cls.onload.record(input_tensor=offloaded, result_tensor=result)
            return result

        return wrapper

    @classmethod
    def track_offload(cls, func: Callable) -> Callable:
        """
        Decorator to track offload operations

        :param func: offload method to decorate
        :return: decorated method
        """

        @functools.wraps(func)
        def wrapper(self, tensor: torch.Tensor | None, *args, **kwargs) -> Any:
            result = func(self, tensor, *args, **kwargs)
            if cls._enabled:
                cls.offload.record(input_tensor=tensor, result_tensor=result)
            return result

        return wrapper

    @classmethod
    def track_update(cls, func: Callable) -> Callable:
        """
        Decorator to track update_offload operations

        :param func: update_offload method to decorate
        :return: decorated method
        """

        @functools.wraps(func)
        def wrapper(self, offloaded: torch.Tensor, data: torch.Tensor | None) -> Any:
            result = func(self, offloaded, data)
            # For updates, the input is the new data and the result is the offloaded tensor
            if cls._enabled:
                cls.update.record(input_tensor=data, result_tensor=offloaded)
            return result

        return wrapper
