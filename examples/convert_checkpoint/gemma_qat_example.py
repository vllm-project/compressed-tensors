# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert the gemma-4 mobile-QAT checkpoints to compressed-tensors.

Produces the released compressed-tensors checkpoints

    google/gemma-4-E2B-it-qat-mobile-transformers -> gemma-4-E2B-it-qat-mobile-ct
    google/gemma-4-E4B-it-qat-mobile-transformers -> gemma-4-E4B-it-qat-mobile-ct

The vision and audio towers stay quantized (vLLM serves them through the
compressed-tensors path), with one exception: three audio-tower linears are
dequantized to dense bf16 because the Transformers Gemma4 audio code inspects
their ``weight.dtype`` in a way that breaks on an integer weight
(``torch.finfo(weight.dtype)`` / ``x.to(dtype=weight.dtype)``):

    ffw_layer_1          (Gemma4AudioFeedForward)
    lconv1d.linear_start (Gemma4AudioLightConv1d)
    self_attn.post       (Gemma4AudioAttention)

Dequantized to bf16, they load through vLLM's unquantized linear fallback.
"""

import json

from compressed_tensors.entrypoints.convert import GemmaConverter, convert_checkpoint

# Audio-tower linears whose dtype the Transformers Gemma4 code reads directly;
# dequantize these to dense bf16 (see module docstring).
DEQUANTIZE_TARGETS = [
    r"(?:^|\.)ffw_layer_1$",
    r"(?:^|\.)lconv1d\.linear_start$",
    r"(?:^|\.)self_attn\.post$",
]

MODEL_IDS = [
    "google/gemma-4-E2B-it-qat-mobile-transformers",
    "google/gemma-4-E4B-it-qat-mobile-transformers",
]

for model_id in MODEL_IDS:
    save_dir = model_id.split("/")[-1].replace("-transformers", "-ct")
    print(f"==> Converting {model_id} -> {save_dir}")

    convert_checkpoint(
        model_stub=model_id,
        save_directory=save_dir,
        converter=GemmaConverter.from_pretrained(
            model_id, dequantize_targets=DEQUANTIZE_TARGETS
        ),
        max_workers=8,
    )

    # The mobile checkpoint omits `architectures`; set the multimodal class so
    # vLLM loads the full text + vision + audio model.
    config_path = f"{save_dir}/config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["architectures"] = ["Gemma4ForConditionalGeneration"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(f"==> Done {save_dir}")
