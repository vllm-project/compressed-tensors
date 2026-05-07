# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    CompressedTensorsDequantizer,
)

# moonshotai/Kimi-K2.6 checkpoint is published in compressed-tensors format.
# This script will upconvert to bfloat16 so that the model can be compressed
# in another configuration.
MODEL_ID = "moonshotai/Kimi-K2.6"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-bf16"

convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=[
            "re:.*mlp.gate$",
            "re:.*lm_head",
            "re:.*embed_tokens$",
            # ignore anything not in language_model
            "re:.*mm_projector.*",
            "re:.*vision.*",
        ],
    ),
    max_workers=1,
)
