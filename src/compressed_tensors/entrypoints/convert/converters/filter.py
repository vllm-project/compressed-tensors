# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Iterable

import torch
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils.match import match_quantizable_tensors


__all__ = ["FilterConverter"]


class FilterConverter(Converter):
    """
    Remove tensors matching target patterns from a checkpoint.
    Useful for stripping auxiliary tensors (e.g. scales, zero points)
    that are no longer needed after dequantization or format conversion.
    """

    def __init__(
        self,
        targets: Iterable[str] = tuple(),
        ignore: Iterable[str] = tuple(),
    ):
        self.targets = list(targets)
        self.ignore = list(ignore)

    def process(self, tensors: dict[str, torch.Tensor]):
        names_to_delete = [
            name
            for _, name in match_quantizable_tensors(
                tensors, self.ignore, self.targets, allow_nonquantizable=True
            )
        ]

        for name in names_to_delete:
            del tensors[name]

    def validate(self, tensors: dict[str, torch.Tensor]):
        pass

    def create_config(self) -> QuantizationConfig | None:
        return None

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()
