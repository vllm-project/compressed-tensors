# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert import AutoAWQConverter, convert_checkpoint
from transformers import AutoConfig


MODEL_ID = "AMead10/Llama-3.2-3B-Instruct-AWQ"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1]

autoawq_config = AutoConfig.from_pretrained(MODEL_ID).quantization_config

# Convert AutoAWQ GEMM tensors into compressed-tensors W4A16 format.
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=AutoAWQConverter.from_autoawq_config(
        autoawq_config,
        quantization_format=CompressionFormat.naive_quantized.value,
    ),
    max_workers=8,
)
