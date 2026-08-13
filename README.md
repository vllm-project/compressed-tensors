# compressed-tensors

The `compressed-tensors` library extends the [safetensors](https://github.com/huggingface/safetensors) format, providing a versatile and efficient way to store and manage compressed tensor data. This library supports various quantization and sparsity schemes, making it a unified format for handling models compressed with algorithms like GPTQ, AWQ, SmoothQuant, and SparseGPT, across formats like INT8, FP8, NVFP4, MXFP4, MXFP8, and more.

## Why `compressed-tensors`?

As model compression becomes increasingly important for efficient deployment of LLMs, the landscape of quantization and compression techniques has become increasingly fragmented.
Each method often comes with its own storage format and loading procedures, making it challenging to work with multiple techniques or switch between them.
`compressed-tensors` addresses this by providing a single, extensible format that can represent a wide variety of compression schemes. 

* **Unified Checkpoint Format**: Supports various compression schemes in a single, consistent format.
* **Wide Compatibility**: Works with popular quantization methods like GPTQ, SmoothQuant, AWQ, AutoRound, etc. See [llm-compressor](https://github.com/vllm-project/llm-compressor)
* **Flexible Quantization Support**: 
  * Activation quantization: W8A8 (int8 and fp8), W4AFP8, Microscale (NVFP4, MXFP4, MXFP8)
  * Mixed precision: W4A16, W8A16, MXFP8A16, MXFP4A16, NVFP4A16
  * Low/arbitrary-bit: WNA4, WNA8, WNA16
  * KV cache quantization: FP8, NVFP4
  * Block quantization (e.g., DeepSeek-style FP8 block)
  * Non-uniform schemes (different layers can be quantized in different ways!)
* **Sparsity Support**: Handles both unstructured and semi-structured (e.g., 2:4) sparsity patterns.
* **Transform Support**: Rotation-based quantization techniques (Hadamard, random Hadamard, random matrix transforms).
* **Checkpoint Conversion**: Convert between formats like AutoAWQ, ModelOpt NVFP4, FP8 block, and compressed-tensors.
* **Model Offloading**: Transparent CPU/disk/multi-GPU offloading for models larger than available VRAM.
* **Open-Source Integration**: Designed to work seamlessly with Hugging Face models, PyTorch, [vLLM](https://github.com/vllm-project/vllm), and [SGLang](https://github.com/sgl-project/sglang).

This allows developers and researchers to easily experiment with composing different quantization methods, simplify model deployment pipelines, and reduce the overhead of supporting multiple compression formats in inference engines.

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

## Getting Started

### Compressing a Model to MXFP4

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

One done, the config.json will have the following quantization_config:
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

See `examples/` for more examples including quantization with calibration (`examples/llama_1.1b/`) and checkpoint conversion (`examples/convert_checkpoint/`).
