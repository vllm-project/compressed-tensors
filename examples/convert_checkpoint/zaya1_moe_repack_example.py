# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    MoEExpertPacker,
)

# Example model with unfused, linearized weights
#   model.layers.0.mlp.experts.0.gate_proj.weight
#   model.layers.0.mlp.experts.0.up_proj.weight
#   model.layers.0.mlp.experts.0.down_proj.weight
MODEL_ID = "inference-optimization/ZAYA1-74B-preview-NVFP4-linear"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-3d"

# Convert 2d linearized weights into 3d packed weights
converter = MoEExpertPacker.from_pretrained(
    model_name_or_path=MODEL_ID,
    fuse_gate_up=True,
)

convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=converter,
    max_workers=8,
)
