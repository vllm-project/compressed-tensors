# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import AutoAWQConverter, convert_checkpoint


MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1]

# Convert AutoAWQ GEMM tensors into compressed-tensors W4A16 format.
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=AutoAWQConverter.from_pretrained(MODEL_ID),
    max_workers=8,
)
