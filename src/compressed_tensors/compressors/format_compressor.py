# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.utils import merge_names
from torch import Tensor
from torch.nn import Module


class FormatCompressor:
    """
    Format-centric classmethod API over registered compressor implementations.
    """

    format: str

    @classmethod
    def _load_impl(cls, config: Any = None) -> BaseCompressor:
        return BaseCompressor.load_from_registry(cls.format, config=config)

    @classmethod
    def compress(
        cls,
        state_dict: dict[str, Tensor],
        *,
        config: Any = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        return cls._load_impl(config=config).compress(state_dict, **kwargs)

    @classmethod
    def decompress(
        cls,
        state_dict_or_path: dict[str, Tensor] | str | Path,
        *,
        config: Any = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        impl = cls._load_impl(config=config)
        if isinstance(state_dict_or_path, dict) and hasattr(impl, "decompress_from_state_dict"):
            values = impl.decompress_from_state_dict(state_dict_or_path, **kwargs)
        else:
            values = impl.decompress(state_dict_or_path, **kwargs)
        return cls._to_state_dict(values)

    @classmethod
    def patch_forward(cls, module: Module):
        raise NotImplementedError()

    @staticmethod
    def _to_state_dict(
        values: Iterable[tuple[str, Tensor | dict[str, Tensor]]],
    ) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for key, value in values:
            if isinstance(value, dict):
                for param_name, param_value in value.items():
                    out[merge_names(key, param_name)] = param_value
            else:
                out[key] = value
        return out
