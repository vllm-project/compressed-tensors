---
title: Installing compressed-tensors
---

# Getting Started

## Installation

### From [PyPI](https://pypi.org/project/compressed-tensors)

Stable release:
```bash
pip install compressed-tensors
```

Nightly release:
```bash
pip install --pre compressed-tensors
```

### From Source

```bash
git clone https://github.com/vllm-project/compressed-tensors
cd compressed-tensors
pip install -e .
```

## Compressing a Model to MXFP4

The following example loads Llama 3 8B, applies round-to-nearest (RTN) MXFP4 weight quantization, compresses the weights, and saves the result. No calibration data is needed — scales are computed directly from the weights.

```python
model_name = "meta-llama/Meta-Llama-3-8B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device, torch_dtype="auto"
)

# Set-up the quantization config. This defines:
# 1. What quantization scheme we're applying and to which layers
# 2. Any layers that should be ignored
# In this case, all the Linear layers are targeted, apart from the lm_head
config = QuantizationConfig(
    config_groups={"MXFP4": ["Linear"]},
    ignore=["lm_head"],
)
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
    # MXFP4 uses group-wise quantization for its weights, with group_size 32
    group_size = args.group_size

    if group_size is not None and group_size > 0:
        reshaped = weight.unflatten(-1, (math.ceil(weight.shape[-1] / group_size), group_size))
        min_val = reshaped.amin(dim=-1)
        max_val = reshaped.amax(dim=-1)
    else:
        min_val, max_val = torch.aminmax(weight)

    # Calculate the quantization parameters, such as the weight scale, using the min and max values
    scale, _ = calculate_qparams(min_val, max_val, args)
    # Update the parameters attached to the module based on the calculated value
    # In this case, we update the `weight_scale` attached to the targeted linear layers
    update_offload_parameter(module, "weight_scale", scale)


output_dir = "./Meta-Llama-3-8B-MXFP4"
# set-up a compressor
compressor = ModelCompressor.from_pretrained_model(model)
# Compress the model using the calibrated scales and save it using the mxfp4-pack-quantized format.
# This format defines the weight packing, which can be seamlessly loaded through vLLM.
compressor.compress_model(model)
model.save_pretrained(output_dir)
# Update the model's config with the relevant compressed-tensors details, illustrated below.
compressor.update_config(output_dir)
```

Once done, the config.json will have the following quantization_config:
```yaml
"quantization_config": {
    "config_groups": {
      "group_0": {
        "format": "mxfp4-pack-quantized",
        "input_activations": {
          "actorder": null,
          "block_structure": null,
          "dynamic": true,
          "group_size": 32,
          "num_bits": 4,
          "observer": null,
          "observer_kwargs": {},
          "scale_dtype": "torch.uint8",
          "strategy": "group",
          "symmetric": true,
          "type": "float",
          "zp_dtype": null
        },
        "output_activations": null,
        "targets": [
          "Linear"
        ],
        "weights": {
          "actorder": null,
          "block_structure": null,
          "dynamic": false,
          "group_size": 32,
          "num_bits": 4,
          "observer": "memoryless_minmax",
          "observer_kwargs": {},
          "scale_dtype": "torch.uint8",
          "strategy": "group",
          "symmetric": true,
          "type": "float",
          "zp_dtype": null
        }
      }
    },
    "format": "mxfp4-pack-quantized",
    "global_compression_ratio": null,
    "ignore": [
      "lm_head"
    ],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {},
    "version": "0.18.1.dev0+gac8e2ba.d20260813"
  },
```

See [`examples/`](https://github.com/vllm-project/compressed-tensors/tree/main/examples) for more examples including quantization with calibration (`examples/llama_1.1b/`) and checkpoint conversion (`examples/convert_checkpoint/`).
