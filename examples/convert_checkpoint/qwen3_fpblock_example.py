# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    FP8BlockToBfloat16Converter,
)

MODEL_ID = "qwen-community/Qwen3-4B-FP8"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1]

# Convert modelopt nvfp4 example to compressed-tensors format
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=FP8BlockToBfloat16Converter(
        # qwen-community/Qwen3-4B-FP8's fp8-block-quantized layers, found by inspection
        targets=[
            "re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
            "re:.*self_attn.*\.(q|k|v|o)_proj$",
        ],
        weight_block_size=[128, 128],
    ),
    max_workers=8,
)
