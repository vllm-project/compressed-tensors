####
#
# The following example shows how a model with pre-calibrated scales
# can be compressed using the nvfp4-pack-quantized compressor.
#
# We define a simple PyTorch model that already includes weight_scale,
# weight_global_scale, and input_global_scale parameters, attach the
# NVFP4 quantization scheme, and then compress.
#
# The nvfp4-pack-quantized format stores:
#   - weight_packed: FP4 weights packed into uint8 (two FP4 values per byte)
#   - weight_scale: per-group scale in float8_e4m3fn (group_size=16)
#   - weight_global_scale: per-tensor scalar (float32) that maps group scales
#     into the FP8 range. Computed as: (fp8_max * fp4_max) / max(abs(weight))
#   - input_global_scale: per-tensor scalar (float32) for input activations
#     (per-group input scales are computed dynamically at runtime)
#
# See: src/compressed_tensors/compressors/nvfp4/base.py
#
####

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
import os
from safetensors.torch import save_file
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import QuantizationScheme, QuantizationStatus

# NVFP4 is the preset quantization args dict for NVFP4 (weights + input activations)
# defined in src/compressed_tensors/quantization/quant_scheme.py
from compressed_tensors.quantization.quant_scheme import NVFP4


GROUP_SIZE = 16

scheme = QuantizationScheme(targets=["Linear"], **NVFP4)


class TinyModel(nn.Module):
    """Simple model with pre-calibrated NVFP4 scales."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128, bias=False)
        self.fc1.weight_scale = nn.Parameter(
            torch.ones(128, 256 // GROUP_SIZE, dtype=torch.float32),
            requires_grad=False,
        )
        self.fc1.weight_global_scale = nn.Parameter(
            torch.tensor([1.0], dtype=torch.float32), requires_grad=False
        )
        self.fc1.input_global_scale = nn.Parameter(
            torch.tensor([1.0], dtype=torch.float32), requires_grad=False
        )

        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc2.weight_scale = nn.Parameter(
            torch.ones(64, 128 // GROUP_SIZE, dtype=torch.float32),
            requires_grad=False,
        )
        self.fc2.weight_global_scale = nn.Parameter(
            torch.tensor([1.0], dtype=torch.float32), requires_grad=False
        )
        self.fc2.input_global_scale = nn.Parameter(
            torch.tensor([1.0], dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        return self.fc2(self.fc1(x))


model = TinyModel()

# Attach the quantization scheme to each layer so the compressor knows
# which modules to compress and how (num_bits, group_size, etc.)
for module in model.modules():
    if isinstance(module, nn.Linear):
        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.FROZEN

# Compress using the nvfp4-pack-quantized compressor - this can also
# just be inferred from the quantization scheme attached to the model or
# overridden with a different quantization_format if desired.
compressor = ModelCompressor.from_pretrained_model(
    model, quantization_format="nvfp4-pack-quantized"
)

# Alternatively, the compressor can be constructed directly:
#   compressor = ModelCompressor(
#       quantization_config=QuantizationConfig(
#           config_groups={"group_1": scheme},
#           quantization_status=QuantizationStatus.FROZEN,
#       ),
#       force_compression_format="nvfp4-pack-quantized",
#   )
compressor.compress_model(model)


output_dir = "./TinyModel-NVFP4"
os.makedirs(output_dir, exist_ok=True)
save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))

# The compressed safetensors file replaces the original weight tensors with:
#
# For each quantized layer (e.g. fc1 with original weight shape [128, 256]):
#   - weight_packed [128, 128] uint8: two FP4 values packed per byte,
#     halving the original weight size
#   - weight_scale [128, 16] float8_e4m3fn: one scale per group of 16 elements
#     (256 / group_size=16 = 16 groups), stored in FP8 to save space
#   - weight_global_scale [1] float32: per-tensor scalar that maps the
#     per-group scales into the full FP8 dynamic range
#   - input_global_scale [1] float32: per-tensor scalar for input activations
#     (per-group input scales are computed dynamically at runtime)
