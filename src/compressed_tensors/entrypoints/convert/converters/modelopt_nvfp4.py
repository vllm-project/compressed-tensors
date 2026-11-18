# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

from compressed_tensors.entrypoints.convert.converters.modelopt import ModelOptConverter
from compressed_tensors.quantization import QuantizationArgs


__all__ = ["ModelOptNvfp4Converter"]


class ModelOptNvfp4Converter(ModelOptConverter):
    """
    Convert params from modelopt NVFP4 to CT NVFP4 convention,
    and optionally the kv_cache_scheme

    Kept for backwards compatibility; ModelOptConverter generalizes this
    converter to every ModelOpt quantization format.
    """

    def __init__(
        self,
        ignore: Iterable[str] = tuple(),
        targets: Iterable[str] = tuple(),
        kv_cache_scheme: QuantizationArgs | None = None,
    ):
        super().__init__(
            quant_algo="NVFP4",
            ignore=ignore,
            targets=targets,
            kv_cache_scheme=kv_cache_scheme,
        )
