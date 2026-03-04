# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class OperationStats:
    """Statistics for a single operation type (onload, offload, or update)"""

    count: int = 0
    bytes_moved: int = 0
    noop_count: int = 0

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
        self.count += 1

        if input_tensor is result_tensor:
            self.noop_count += 1
        elif result_tensor is not None:
            self.bytes_moved += result_tensor.element_size() * result_tensor.numel()

    @staticmethod
    def _is_noop(
        input_tensor: torch.Tensor | None,
        result_tensor: torch.Tensor | None,
    ) -> bool:
        """
        Determine if an operation was a no-op (no actual data movement)

        :param input_tensor: input tensor
        :param result_tensor: result tensor
        :return: True if the operation was a no-op
        """
        # None tensors are always no-ops
        if input_tensor is None or result_tensor is None:
            return True

        # Check if data pointer and device are the same (no actual movement)
        return (
            result_tensor.data_ptr() == input_tensor.data_ptr()
            and result_tensor.device == input_tensor.device
        )


class OffloadStats:
    """
    Global statistics tracker for OffloadCache operations.

    This class should never be instantiated. All methods are class methods and all
    statistics are stored as class variables.

    Tracks the number of onload, offload, and update operations, as well as
    the number of bytes affected and whether operations were no-ops.

    Example usage:
        # Statistics are collected automatically when decorators are applied
        stats = OffloadStats.get_stats()
        print(stats)

        # Reset statistics
        OffloadStats.reset()

        # Get formatted summary
        summary = OffloadStats.format_summary()
        print(summary)
    """

    # Class-level statistics
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
    def reset(cls):
        """Reset all statistics to zero"""
        cls.onload = OperationStats()
        cls.offload = OperationStats()
        cls.update = OperationStats()

    @classmethod
    def format_summary(cls, unit: str = "MB") -> str:
        """
        Generate a formatted summary of device movement statistics

        :param unit: unit for displaying bytes ('B', 'KB', 'MB', or 'GB')
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
            cls.update.record(input_tensor=data, result_tensor=offloaded)
            return result

        return wrapper
