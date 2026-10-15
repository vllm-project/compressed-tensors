# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import HummingConverter, convert_checkpoint


MODEL_ID = "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-ct"

# Convert a GSQ ("humming") checkpoint into compressed-tensors pack-quantized.
# Renames packed `.weight` -> `.weight_packed`, emits `.weight_shape`, and rewrites
# `quantization_config` so vLLM's compressed-tensors loader can consume it.
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=HummingConverter.from_pretrained(MODEL_ID),
    max_workers=8,
)
