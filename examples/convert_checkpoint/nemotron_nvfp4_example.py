# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    ModelOptConverter,
    convert_checkpoint,
)

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1]

# Convert the ModelOpt checkpoint to compressed-tensors format. The converter
# reads the checkpoint's ModelOpt quantization config, so targets, the ignore
# list, the kv cache scheme and the per-module quant_algo (this model mixes
# NVFP4 experts with FP8 dense layers) are all derived from it
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=ModelOptConverter.from_pretrained(MODEL_ID),
)
