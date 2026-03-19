# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    reindex_checkpoint,
    FP8BlockToBfloat16Converter,
)

# deepseek-ai/DeepSeek-V3.2 checkpoint has layers that are quantized in the FP8
# quant method's FP8_BLOCK scheme. This script will upconvert to bfloat16 so that
# the model can be compressed in another configuration.
MODEL_ID = "deepseek-ai/DeepSeek-V3.2"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-bf16"

converter = FP8BlockToBfloat16Converter(
    # `deepseek-ai/DeepSeek-V3.2` fp8-block-quantized layers, found by inspection
    targets=[
        r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
        r"re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj$",
        r"re:.*self_attn.kv_a_proj_with_mqa$",
        r"re:.*self_attn.indexer.(wk|wq_b)$",
    ],
)

# A module's `weight` and `weight_scale_inv` tensors need to exist in the same
# safetensors file in order for convert_checkpoint to work in a parallelized fashion.
# First, reindex weight and weight_scale_inv tensors if they are split across files
reindex_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    get_unmatched_names=converter.get_unmatched_names,
)
# Convert DeepSeek-V3.2 back to dense bfloat16 format
convert_checkpoint(
    model_stub=SAVE_DIR,
    save_directory=SAVE_DIR,
    converter=converter,
    max_workers=4,
)
