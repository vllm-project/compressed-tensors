# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    FP8Converter,
)

# deepseek-ai/DeepSeek-V4.1-Flash-1017 checkpoint is quantized with the
# transformers "fp8" quant method's mixed-precision scheme: linear weights are
# block-quantized to fp8 (with ue8m0 scales) while MoE expert weights are
# packed as fp4. This script converts both into the equivalent
# compressed-tensors format. The converted checkpoint is saved in the same
# folder as the HuggingFace hub cache.
MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-CT"

# Convert DeepSeek-V4.1-Flash-1017 into compressed-tensors format. The
# converter reads weight_block_size and modules_to_not_convert from the model
# config, and inspects the checkpoint headers to split modules into fp8 vs
# fp4 targets.
#
# The conversion runs on CPU: two shards each hold a ~100 GB engram embedding
# table ([384006168, 256] fp8), which does not fit on a single GPU, while host
# RAM can buffer each shard comfortably.
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=FP8Converter.from_pretrained(MODEL_ID),
    device="cpu",
    max_workers=8,
)
