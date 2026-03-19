# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    reindex_checkpoint,
    FP8BlockToBfloat16Converter,
)

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
    weight_block_size=[128, 128],
)

# Some weight and weight_scale_inv tensors are split across safetensors files
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
