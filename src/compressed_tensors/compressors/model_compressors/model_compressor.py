# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
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
from compressed_tensors.config.format import flatten_formats, infer_set_module_formats
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationScheme,
)
from compressed_tensors.transform import TransformConfig
from loguru import logger
from tqdm import tqdm
from transformers import CompressedTensorsConfig
from transformers.file_utils import CONFIG_NAME


__all__ = ["ModelCompressor"]


class ModelCompressor:
    """
    Orchestrates compression and decompression of a quantized model.

    Compression Lifecycle
        - compressor = ModelCompressor.from_pretrained_model(model)
        - compressor.compress_model(model)
        - model.save_pretrained(quantized_path)
        - compressor.update_config(quantized_path)

    Decompression Lifecycle
        - model = AutoModelForCausalLM.from_pretrained(quantized_path)
            - CompressedTensorsHfQuantizer.__init__
                - compressor = ModelCompressor.from_compression_config(ct_config)
            - CompressedTensorsHfQuantizer._process_model_before_weight_loading
                - apply_quantization_config(model, ct_config.quantization_config)
                - compressor.compress_model(model)
            - CompressedTensorsHfQuantizer._process_model_after_weight_loading
                - if run_compressed == False: compressor.decompress_model(model)
    """

    # these attributes are used by `CompressedTensorsHfQuantizer` to apply configs
    # during `_process_model_before_weight_loading`
    quantization_config: QuantizationConfig | None
    transform_config: TransformConfig | None
    force_compression_format: str | None

    @classmethod
    def from_compression_config(cls, compression_config: CompressedTensorsConfig):
        """
        HFQuantizer uses this entrypoint to load quantized models by passing an
        instance of `CompressedTensorsConfig`

        :param compression_config: instance of `CompressedTensorsConfig` containing
            quantized model information
        """
        # vLLM Cutlass24 implementation is no longer supported
        if not isinstance(compression_config, CompressedTensorsConfig):
            raise ValueError(
                f"Support for compression config of type {type(compression_config)} "
                "is no longer supported. If you are attempting to use a Sparse24 "
                "model, note that the Sparse24 format is not longer supported by "
                "as of `compressed-tensors>0.14.0`"
            )

        # transform_config is added by transformers#42887
        q_config = compression_config.quantization_config
        t_config = getattr(compression_config, "transform_config", None)

        return cls(quantization_config=q_config, transform_config=t_config)

    @classmethod
    def from_pretrained_model(
        cls,
        model: torch.nn.Module,
        sparsity_config_or_format: Optional[SparsityCompressionConfig] = None,
        quantization_format: Optional[str] = None,
    ):
        """
        LLM Compressor uses this entrypoint to compress models before saving
        Given a path to a model config, extract the sparsity and/or quantization
        configs and load a ModelCompressor.

        Note: passing `sparsity_config_or_format` is no longer supported

        :param model: model with quantization config applied
        :param quantization_format: string corresponding to a quantization format
            that should be applied to the entire model, overrides inferred formats
            for all quantized modules
        """
        if sparsity_config_or_format is not None:
            logger.warning(
                "Passing sparsity config or format to `ModelCompressor` is no longer "
                "supported."
            )

        force_compression_format = quantization_format
        if force_compression_format == CompressionFormat.dense.value:
            force_compression_format = None

        # reconstruct qconfig from qschemes that are attached to the model
        quantization_config = QuantizationConfig.from_pretrained(model)
        transform_config = getattr(model, TRANSFORM_CONFIG_NAME, None)

        return cls(
            quantization_config=quantization_config,
            transform_config=transform_config,
            force_compression_format=force_compression_format,
        )

    def __init__(
        self,
        quantization_config: Optional[QuantizationConfig] = None,
        transform_config: Optional[TransformConfig] = None,
        force_compression_format: Optional[str] = None,
    ):
        self.quantization_config = quantization_config
        self.transform_config = transform_config
        self.force_compression_format = force_compression_format

    def compress_model(self, model: torch.nn.Module):
        """
        Compress the model's parameters in memory.

        Supports ``torch.distributed``: each rank compresses its own shard of
        modules (round-robin by sorted module names) and then all compressed
        parameters are broadcast to the remaining ranks.

        :param model: model whose parameters should be compressed in place
        """
        # infer and set which format each module should be compressed with
        formats = infer_set_module_formats(model, self.force_compression_format)

        # update quantization config format for better UX when reading config.json
        if self.quantization_config is not None:
            self.quantization_config.format = flatten_formats(formats)

        # compress modules
        modules = model.named_modules(remove_duplicate=True)
        for _, module in tqdm(list(modules), desc="Compressing model"):
            scheme = getattr(module, "quantization_scheme", None)
            if not isinstance(scheme, QuantizationScheme):
                continue

            assert scheme.format is not None, "Must set format before decompressing"
            compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
            compressor.compress_module(module)

        # attempting to perform forward passes with a compressed model
        # will cause to the model to decompress. This allows for generation
        # without requiring CT to handle quantized kernels
        self.add_decompress_hook(model)

    def decompress_model(self, model: torch.nn.Module):
        """
        Decompress the model's parameters in memory.

        Supports ``torch.distributed``: mirrors the ``compress_model`` structure
        with rank-sharded work and a broadcast step at the end.

        :param model: model whose parameters should be decompressed in place
        """
        # decompress modules
        modules = model.named_modules(remove_duplicate=True)
        for _, module in tqdm(list(modules), desc="Decompressing model"):
            scheme = getattr(module, "quantization_scheme", None)
            if not isinstance(scheme, QuantizationScheme):
                continue

            assert scheme.format is not None, "Must set format before decompressing"
            compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
            compressor.decompress_module(module)

    def update_config(self, save_directory: str):
        """
        Update the model config located at save_directory with compression configs

        :param save_directory: path to a folder containing a HF model config
        """
        if not any((self.quantization_config, self.transform_config)):
            return

        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        qconfig_data = (
            self.quantization_config.model_dump(exclude=["quant_method"])
            if self.quantization_config is not None
            else {}
        )
        tconfig_data = (
            self.transform_config.model_dump()
            if self.transform_config is not None
            else {}
        )

        config_data[QUANTIZATION_CONFIG_NAME] = {
            COMPRESSION_VERSION_NAME: compressed_tensors.__version__,
            QUANTIZATION_METHOD_NAME: DEFAULT_QUANTIZATION_METHOD,
            SPARSITY_CONFIG_NAME: {},  # sparsity is removed for now
            TRANSFORM_CONFIG_NAME: tconfig_data,
            **qconfig_data,
        }

        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)

    def add_decompress_hook(self, model: torch.nn.Module):
        def decompress_hook(model, args):
            self.decompress_model(model)
            model.decompress_hook.remove()

        model.decompress_hook = model.register_forward_pre_hook(decompress_hook)
