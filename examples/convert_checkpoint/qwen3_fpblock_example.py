# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    FP8BlockDequantizer,
)

MODEL_ID = "qwen-community/Qwen3-4B-FP8"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1].rstrip("-FP8")

# Convert Qwen3-4B-FP8 back to dense bfloat16 format
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=FP8BlockDequantizer(
        # qwen-community/Qwen3-4B-FP8's fp8-block-quantized layers, found by inspection
        targets=[
            r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
            r"re:.*self_attn.*\.(q|k|v|o)_proj$",
        ],
        weight_block_size=[128, 128],
    ),
    max_workers=8,
)
