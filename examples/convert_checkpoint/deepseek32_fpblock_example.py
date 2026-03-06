# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    FP8BlockToBfloat16Converter,
)
from llmcompressor.entrypoints.model_free import reindex_fused_weights

MODEL_ID = "deepseek-ai/DeepSeek-V3.2"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-bf16"

# Some weight and weight_scale_inv tensors are split across safetensors files
reindex_fused_weights(model_stub=MODEL_ID, save_directory=SAVE_DIR)
# Convert Qwen3-4B-FP8 back to dense bfloat16 format
# convert_checkpoint(
#     model_stub=SAVE_DIR,
#     save_directory=SAVE_DIR,
#     converter=FP8BlockToBfloat16Converter(
#         # qwen-community/Qwen3-4B-FP8's fp8-block-quantized layers, found by inspection
#         targets=[
#             "re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
#             "re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj$",
#             "re:.*self_attn.kv_a_proj_with_mqa$",
#             "re:.*self_attn.indexer.(wk|wq_b)$",
#         ],
#         weight_block_size=[128, 128],
#     ),
#     max_workers=1,
# )
