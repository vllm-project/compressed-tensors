# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import compressed_tensors
import torch
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.config.base import SparsityStructure
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.transform import TransformConfig
from compressed_tensors.utils import replace_module
from torch.nn import Module
from tqdm import tqdm
from transformers import CompressedTensorsConfig
from transformers.file_utils import CONFIG_NAME


__all__ = ["ModelCompressor"]


class ModelCompressor:
    """
    Handles compression and decompression of a model with a sparsity config and/or
    quantization config.

    Compression LifeCycle
        - compressor = ModelCompressor.from_pretrained_model(model)
        - compressed_state_dict = compressor.compress(model, state_dict)
            - compressor.quantization_compressor.compress(model, state_dict)
            - compressor.sparsity_compressor.compress(model, state_dict)
        - model.save_pretrained(output_dir, state_dict=compressed_state_dict)
        - compressor.update_config(output_dir)

    Decompression LifeCycle
        - compressor = ModelCompressor.from_pretrained(comp_model_path)
        - model = AutoModel.from_pretrained(comp_model_path)
        - compressor.decompress(comp_model_path, model)
            - compressor.sparsity_compressor.decompress(comp_model_path, model)
            - compressor.quantization_compressor.decompress(comp_model_path, model)

    :param sparsity_config: config specifying sparsity compression parameters
    :param quantization_config: config specifying quantization compression parameters
    """

    sparsity_config: Optional[SparsityCompressionConfig] = None
    quantization_config: Optional[QuantizationConfig] = None
    transform_config: Optional[TransformConfig] = None

    @classmethod
    def from_compression_config(cls, compression_config: CompressedTensorsConfig):
        """
        Initialize model compressor from a quantization config in `config.json`

        Entrypoint used by HFQuantizer

        :param compression_config: instance of `CompressedTensorsConfig`
        :return: compressor for the configs, or None if model is not compressed
        """
        assert isinstance(compression_config, CompressedTensorsConfig)

        return cls(
            quantization_config=compression_config.quantization_config,
            sparsity_config=compression_config.sparsity_config,
            # Note: transform_config is not yet supported by `CompressedTensorsConfig`
        )

    @classmethod
    def from_pretrained_model(
        cls,
        model: Module,
        sparsity_config_or_format: Optional[
            SparsityCompressionConfig | SparsityStructure
        ] = None,
        quantization_format: Optional[CompressionFormat] = None,
    ):
        """
        Initialize model compressor from model with "initialized" or "frozen"
        quantization status.

        Entrypoint used by LLM Compressor during model saving

        :param model: pytorch model to target for compression
        :param sparsity_config_or_format: override sparsity format
        :param quantization_format: override quantization format
        :return: compressor for the configs, or None if model is not compressed
        """
        # Quantization Config: reconstruct from attached schemes
        quantization_config = QuantizationConfig.from_pretrained(
            model, format=quantization_format
        )

        # Sparsity Config: passed by caller
        if isinstance(sparsity_config_or_format, str):
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config_or_format
            )
        else:
            sparsity_config = sparsity_config_or_format

        # Transform Config: attached to model
        transform_config = getattr(model, TRANSFORM_CONFIG_NAME, None)

        return cls(
            sparsity_config=sparsity_config,
            quantization_config=quantization_config,
            transform_config=transform_config,
            compression_formats=quantization_format,
        )

    def __init__(
        self,
        sparsity_config: Optional[SparsityCompressionConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        transform_config: Optional[TransformConfig] = None,
    ):
        self.sparsity_config = sparsity_config
        self.quantization_config = quantization_config
        self.transform_config = transform_config

    # ----- model memory compression/decompression pathways ----- #

    def compress_model(self, model: Module):
        """
        Compress model which has an "initialized" or "frozen" quantization status. This
        means that `apply_quantization_config` has already been applied to the model
        prior to calling this function.

        :param model: model containing parameters to compress
        """

        def compress_module(name: str, module: torch.nn.Module):
            scheme: Optional[QuantizationScheme] = getattr(
                module, "quantization_scheme", None
            )
            status: Optional[QuantizationStatus] = getattr(
                module, "quantization_status", None
            )
            if (
                scheme is None
                or status is None
                or status >= QuantizationStatus.COMPRESSED
            ):
                return

            compressor = BaseCompressor.get_value_from_registry(scheme.format)
            module = compressor.compress_module(module)
            setattr(module, "quantization_status", QuantizationStatus.COMPRESSED)
            replace_module(name, module)

        with ProcessPoolExecutor(max_workers=None) as executor:
            modules = model.named_modules(remove_duplicate=True)
            futures = executor.map(compress_module, modules)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Compressing model"
            ):
                pass

    def decompress_model(self, model: Module):
        """
        Decompress a model which has the "compressed" quantization status

        :param model: model containing parameters to decompress
        """

        def decompress_module(name: str, module: torch.nn.Module):
            scheme: Optional[QuantizationScheme] = getattr(
                module, "quantization_scheme", None
            )
            status: Optional[QuantizationStatus] = getattr(
                module, "quantization_status", None
            )
            if (
                scheme is None
                or status is None
                or status < QuantizationStatus.COMPRESSED
            ):
                return

            compressor = BaseCompressor.get_value_from_registry(scheme.format)
            module = compressor.decompress_module(module)
            setattr(module, "quantization_status", QuantizationStatus.FROZEN)
            replace_module(name, module)

        with ProcessPoolExecutor(max_workers=None) as executor:
            modules = model.named_modules(remove_duplicate=True)
            futures = executor.map(decompress_module, modules)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Decompressing model"
            ):
                pass

    def update_config(self, save_directory: str):
        """
        Update the model config located at save_directory with compression configs
        for sparsity and/or quantization

        :param save_directory: path to a folder containing a HF model config
        """
        # this check is also done in `from_pretrained_model`,
        # but not in `from_pretrained`` or `from_compression_config``
        if not any(
            (self.quantization_config, self.sparsity_config, self.transform_config)
        ):
            return

        # write to config.json file, regardless of whether it exists already
        # overwrite previous config and version if already existing
        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        # serialize configs into json
        qconfig_data = (
            self.quantization_config.model_dump(exclude=["quant_method"])
            if self.quantization_config is not None
            else {}
        )
        sconfig_data = (
            self.sparsity_config.model_dump()
            if self.sparsity_config is not None
            else {}
        )
        tconfig_data = (
            self.transform_config.model_dump()
            if self.transform_config is not None
            else {}
        )

        # construct compression (quantization) config
        config_data[QUANTIZATION_CONFIG_NAME] = {
            COMPRESSION_VERSION_NAME: compressed_tensors.__version__,
            QUANTIZATION_METHOD_NAME: DEFAULT_QUANTIZATION_METHOD,
            SPARSITY_CONFIG_NAME: sconfig_data,
            TRANSFORM_CONFIG_NAME: tconfig_data,
            **qconfig_data,
        }

        # write results to config.json file
        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)
