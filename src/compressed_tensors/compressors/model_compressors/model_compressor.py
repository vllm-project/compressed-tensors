# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from functools import partial
from typing import Optional

import compressed_tensors
import torch
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.compressors.distributed import apply_module_parallel
from compressed_tensors.compressors.format import infer_model_format
from compressed_tensors.config import CompressionFormat
from compressed_tensors.offload import is_distributed
from compressed_tensors.offload.utils import module_size
from compressed_tensors.quantization import QuantizationConfig, QuantizationStatus
from compressed_tensors.quantization.utils.helpers import is_module_quantized
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
        - model.save_pretrained_wrapper(quantized_path)
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
    force_compression_format: CompressionFormat | None

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
        sparsity_config_or_format: Optional[object] = None,
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
            logger.warning("Passing sparsity config or format is no longer supported")

        # reconstruct qconfig from qschemes that are attached to the model
        quantization_config = QuantizationConfig.from_pretrained(model)
        transform_config = getattr(model, TRANSFORM_CONFIG_NAME, None)

        # update quantization config format for better UX when reading config.json
        if quantization_config is not None:
            quantization_config.format = infer_model_format(model, quantization_format)

        return cls(
            quantization_config=quantization_config,
            transform_config=transform_config,
            force_compression_format=quantization_format,
        )

    def __init__(
        self,
        quantization_config: Optional[QuantizationConfig] = None,
        transform_config: Optional[TransformConfig] = None,
        force_compression_format: Optional[str] = None,
    ):
        self.quantization_config = quantization_config
        self.transform_config = transform_config
        self.force_compression_format = (
            CompressionFormat(force_compression_format)
            if force_compression_format is not None
            else None
        )

    def compress_model(self, model: torch.nn.Module) -> None:
        """
        Compress the model's parameters in memory

        :param model: model whose parameters should be compressed in place
        """
        # Get all quantized modules
        quantized_modules = [
            module
            for _, module in model.named_modules(remove_duplicate=True)
            if is_module_quantized(module)
        ]

        # Compress modules using distributed or sequential approach
        if is_distributed():
            apply_fn = partial(
                compress_module, force_compression_format=self.force_compression_format
            )
            apply_module_parallel(quantized_modules, apply_fn, module_size)
        else:
            for module in tqdm(quantized_modules, desc="Compressing model"):
                compress_module(module, self.force_compression_format)

        # update config status to reflect compression
        if self.quantization_config is not None:
            self.quantization_config.quantization_status = QuantizationStatus.COMPRESSED

        # attempting to perform forward passes with a compressed model
        # will cause to the model to decompress. This allows for generation
        # without requiring CT to handle quantized kernels
        self.add_decompress_hook(model)

    def decompress_model(self, model: torch.nn.Module) -> None:
        """
        Decompress the model's parameters in memory

        :param model: model whose parameters should be decompressed in place
        """
        # decompress modules
        modules = model.named_modules(remove_duplicate=True)
        for _, module in tqdm(list(modules), desc="Decompressing model"):
            if is_module_quantized(module):
                decompress_module(module, self.force_compression_format)

        # update config status to reflect decompression
        if self.quantization_config is not None:
            self.quantization_config.quantization_status = (
                QuantizationStatus.DECOMPRESSED
            )

        # decompression hook is no longer necessary
        self.remove_decompression_hook(model)

    def update_config(self, save_directory: str) -> None:
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
            QUANTIZATION_METHOD_NAME: QUANTIZATION_METHOD,
            SPARSITY_CONFIG_NAME: {},  # sparsity is removed for now
            TRANSFORM_CONFIG_NAME: tconfig_data,
            **qconfig_data,
        }

        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)

    def add_decompress_hook(self, model: torch.nn.Module):
        """
        Register a forward pre-hook that decompresses the model on first forward pass.

        The hook automatically removes itself after decompression, allowing the model
        to run in decompressed state for inference.

        :param model: model to attach the decompression hook to
        """

        def ct_decompress_hook(model, args):
            self.decompress_model(model)
            # decompress_model already removes the hook via remove_decompression_hook

        model.ct_decompress_hook = model.register_forward_pre_hook(ct_decompress_hook)

    def remove_decompression_hook(self, model: torch.nn.Module):
        """
        Remove the decompression hook from the model if it exists.

        Called after manual decompression to clean up the hook that would
        otherwise trigger on the next forward pass.

        :param model: model to remove the decompression hook from
        """
        if hasattr(model, "ct_decompress_hook"):
            model.ct_decompress_hook.remove()
            delattr(model, "ct_decompress_hook")
