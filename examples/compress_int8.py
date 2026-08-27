####
#
# The following example shows how a model can be calibrated and
# compressed entirely with primitives within `compressed-tensors`
# using PyTorch hooks.
# The int8_config.json defines W8A8 symmetric quantization
# (INT8 weights + INT8 activations), so we need calibration data
# to compute both weight and input activation scales.
# Note: This is doing basic quantization. For better accuracy recovery,
# consider using GPTQ through LLM-Compressor
#
####

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationConfig,
    apply_quantization_config,
)
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator


# Set-up the quantization config from a JSON file. This defines:
# 1. What quantization scheme we're applying and to which layers
# 2. Any layers that should be ignored
# In this case, int8_config.json targets all Linear layers with W8A8 INT8 quantization.
# All the quantization arguments are described through the QuantizationArgs,
# found in src/compressed_tensors/quantization/quant_args.py
config_file = Path(__file__).parent / "int8_config.json"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
output_dir = "./Meta-Llama-3-8B-Instruct-W8A8"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

config = QuantizationConfig.model_validate_json(config_file.read_text())


# Apply the config to the model. This step uses the config to define
# the quantization parameters (such as the scales) for the targeted layers
# and attaches a QuantizationScheme which defines how the weights and activations
# should be quantized (e.g number of bits, group or block sizes, etc)
apply_quantization_config(model, config)


# Register a forward hook to compute scales during calibration.
# Since the config is symmetric, we only need scales (no zero points).
# On each forward pass, the hook computes min/max of the weight tensor and
# input activation tensor, then derives the quantization scale via
# calculate_qparams (src/compressed_tensors/quantization/utils/helpers.py)
def update_scales_hook(
    module: torch.nn.Module, input: torch.Tensor, _output: torch.Tensor
):
    from compressed_tensors.quantization.utils import calculate_qparams
    from compressed_tensors.offload import update_offload_parameter

    quantization_scheme = getattr(module, "quantization_scheme", None)
    if not quantization_scheme:
        return

    quantization_args = getattr(quantization_scheme, "weights", None)
    min_val, max_val = torch.aminmax(module.weight.data)
    scale, _ = calculate_qparams(min_val, max_val, quantization_args)
    update_offload_parameter(module, "weight_scale", scale)

    quantization_args = getattr(quantization_scheme, "input_activations", None)
    min_val, max_val = torch.aminmax(input[0])
    scale, _ = calculate_qparams(min_val, max_val, quantization_args)
    update_offload_parameter(module, "input_scale", scale)


model.apply(lambda module: module.register_forward_hook(update_scales_hook))

# Load and preprocess calibration dataset
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


tokenized_dataset = ds.map(tokenize, remove_columns=ds.column_names)
data_loader = DataLoader(
    tokenized_dataset,
    batch_size=1,
    collate_fn=DefaultDataCollator(),
    sampler=RandomSampler(tokenized_dataset),
)

# Run calibration - the hook will update weight and input activation scales
# on each forward pass for every module with a quantization_scheme attached
with torch.no_grad():
    for idx, sample in tqdm(enumerate(data_loader), desc="Running calibration"):
        sample = {k: v.to(model.device) for k, v in sample.items()}
        _ = model(**sample)


# Set up a compressor.
# The compression format is inferred by iterating over all quantized modules and
# checking each format's `can_compress(module_type, scheme)` in priority order
# (see compressed_tensors/compressors/format.py). The format can also be set
# explicitly by passing quantization_format (e.g. quantization_format="int-quantized")
compressor = ModelCompressor.from_pretrained_model(model)

# Compress the model using the calibrated scales
compressor.compress_model(model)

# Save the compressed model and update config with quantization details
model.save_pretrained(output_dir)
compressor.update_config(output_dir)
