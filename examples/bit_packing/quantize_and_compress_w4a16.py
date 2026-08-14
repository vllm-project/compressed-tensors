####
#
# The following example shows how a model can be quantized and packed
# entirely with primitives within `compressed-tensors`.
# The int4_config.json defines weights-only INT4 quantization,
# so we only need to compute weight scales (no calibration data needed).
# Weights are packed into int32 using the PackedQuantizationCompressor:
# src/compressed_tensors/compressors/pack_quantized/base.py
# Note: This is doing basic quantization. For better accuracy recovery, 
# consider using GPTQ throught LLM-Compressor
#
####

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from pathlib import Path

import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.offload import update_offload_parameter
from compressed_tensors.quantization import (
    QuantizationConfig,
    apply_quantization_config,
)
from compressed_tensors.quantization.utils import calculate_qparams
from transformers import AutoModelForCausalLM


model_name = "meta-llama/Meta-Llama-3-8B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device, torch_dtype="auto"
)



# Set-up the quantization config from a JSON file. This defines:
# 1. What quantization scheme we're applying and to which layers
# 2. Any layers that should be ignored
# In this case, int4_config.json targets all Linear layers with weights-only INT4 quantization.
# All the quantization arguments are described through the QuantizationArgs, 
# found in src/compressed_tensors/quantization/quant_args.py
 
config_file = Path(__file__).parent / "int4_config.json"
config = QuantizationConfig.model_validate_json(config_file.read_text())

# Apply the config to the model. This step uses the config to define
# the quantization parameters (such as the scales) for the targeted layers
# and attaches a QuantizationScheme which defines how the weights and activations
# should be quantized (e.g number of bits, group or block sizes, etc)
apply_quantization_config(model, config)

# Compute weight scales using round-to-nearest quantization
for name, module in model.named_modules():
    # Only target layers with a QuantizationScheme attached
    scheme = getattr(module, "quantization_scheme", None)
    if scheme is None or scheme.weights is None:
        continue

    weight = module.weight.data
    args = scheme.weights
    # Read the group_size defined in the user provided recipe 
    # In this case, that is group_size=128 defined in int4_config.json
    group_size = args.group_size

    # Reshape based on the group_size. With group quantization, we apply a scale every
    # group_size number of rows. Reshape to make it easier to identify the min and max values per group.
    if group_size is not None and group_size > 0:
        reshaped = weight.unflatten(
            -1, (math.ceil(weight.shape[-1] / group_size), group_size)
        )
        min_val = reshaped.amin(dim=-1)
        max_val = reshaped.amax(dim=-1)
    else:
        min_val, max_val = torch.aminmax(weight)

    # Calculate the quantization parameters, such as the weight scale, using the min and max values
    scale, _ = calculate_qparams(min_val, max_val, args)
    # Update the parameters attached to the module based on the calculated value
    # In this case, we update the `weight_scale` attached to the targeted linear layers
    update_offload_parameter(module, "weight_scale", scale)

# set-up a compressor
# The compression format is inferred by iterating over all quantized modules and
# checking each format's `can_compress(module_type, scheme)` in priority order
# (see compressed_tensors/compressors/format.py). The format can also be set
# explicitly by passing quantization_format (e.g. quantization_format="pack-quantized")
compressor = ModelCompressor.from_pretrained_model(model)
# Compress the model using the calibrated scales and save it using the pack-quantized format.
# This format defines the weight packing, which can be seamlessly loaded through vLLM.
compressor.compress_model(model)

output_dir = "./Meta-Llama-3-8B-W4A16"
# With pack-quantized, weights are quantized and then packed into int32 tensors
# (e.g. eight 4-bit values per int32). The saved state dict stores weight_packed,
# weight_scale, and weight_shape (replacing the original weight tensor).
model.save_pretrained(output_dir)
# Update the model's config with the relevant compressed-tensors details
compressor.update_config(output_dir)
