# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert_checkpoint import (
    convert_checkpoint,
    ModelOptNvfp4Converter,
)

MODEL_ID = "nvidia/Qwen3-32B-NVFP4"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1]

# Convert modelopt nvfp4 example to compressed-tensors format
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converters=[
        ModelOptNvfp4Converter(
            # nvidia/Qwen3-32B-NVFP4's nvfp4-quantized layers, found by inspection
            targets=[
                "re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
                "re:.*self_attn.*\.(q|k|v|o)_proj$",
            ]
        )
    ],
    max_workers=8,
)
