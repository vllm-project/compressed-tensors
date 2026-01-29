####
#
# The following example shows how the example in `ex_config_quantization.py`
# can be done within vllm's llm-compressor project
# Be sure to `pip install llmcompressor` before running
# See https://github.com/vllm-project/llm-compressor for more information
#
####

from pathlib import Path

import torch
from llmcompressor import oneshot


recipe = str(Path(__file__).parent / "example_quant_recipe.yaml")
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_name = "open_platypus"
split = "train"
num_calibration_samples = 512
max_seq_length = 1024
pad_to_max_length = False
output_dir = "./llama1.1b_llmcompressor_quant_out"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

oneshot(
    model=model_name,
    dataset=dataset_name,
    output_dir=output_dir,
    overwrite_output_dir=True,
    max_seq_length=max_seq_length,
    num_calibration_samples=num_calibration_samples,
    recipe=recipe,
    pad_to_max_length=pad_to_max_length,
)
