# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    FP8BlockToBfloat16Converter,
)
from compressed_tensors.modeling.deepseekv32.model import DeepseekV32ForCausalLM
from compressed_tensors.modeling.deepseekv32.config import (
    ModelConfig as Deepseek32Config,
)

from llmcompressor.entrypoints.model_free.reindex_fused_weights import (
    reindex_fused_weights,
)
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("deepseek_v32", Deepseek32Config)
AutoModelForCausalLM.register(Deepseek32Config, DeepseekV32ForCausalLM)


MODEL_ID = "deepseek-ai/DeepSeek-V3.2"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-bf16"

# Some weight and weight_scale_inv tensors are split across safetensors files
reindex_fused_weights(model_stub=MODEL_ID, save_directory=SAVE_DIR)
# Convert DeepSeek-V3.2 back to dense bfloat16 format
convert_checkpoint(
    model_stub=SAVE_DIR,
    save_directory=SAVE_DIR,
    converter=FP8BlockToBfloat16Converter(
        # DeepSeek-V3.2's fp8-block-quantized layers, found by inspection
        targets=[
            r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
            r"re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj$",
            r"re:.*self_attn.kv_a_proj_with_mqa$",
            r"re:.*self_attn.indexer.(wk|wq_b)$",
        ],
        weight_block_size=[128, 128],
    ),
    max_workers=4,
)
