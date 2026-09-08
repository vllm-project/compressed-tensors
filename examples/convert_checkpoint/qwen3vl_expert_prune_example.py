# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    MagnitudeExpertPruner,
)

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-expert-pruned"

# Prune half of the MoE experts by router-weight magnitude.
# Patterns are scoped to the language_model so the vision tower is left untouched.
# Qwen3-VL-MoE stores experts as 3D stacked tensors (experts.gate_up_proj /
# experts.down_proj) and the router as mlp.gate.weight, found by inspection.
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=MagnitudeExpertPruner.from_pretrained(
        MODEL_ID,
        router_pattern=r"language_model\..*\.mlp\.gate\.weight$",
        expert_pattern=r"language_model\..*\.mlp\.experts\.(gate_up_proj|down_proj)$",
        sparsity=0.5,
    ),
    max_workers=8,
)
