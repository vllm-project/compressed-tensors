---
title: Home
---

# compressed-tensors

The `compressed-tensors` library extends the [safetensors](https://github.com/huggingface/safetensors) format, providing a versatile and efficient way to store and manage compressed tensor data. This library supports various compression schemes, making it a unified format for handling models compressed with algorithms like GPTQ, AWQ, SmoothQuant, and SparseGPT, across formats like INT8, FP8, NVFP4, MXFP4, MXFP8, and more.

## Why `compressed-tensors`?

As model compression becomes increasingly important for efficient deployment of LLMs, the landscape of quantization and compression techniques has become increasingly fragmented.
Each method often comes with its own storage format and loading procedures, making it challenging to work with multiple techniques or switch between them.
`compressed-tensors` addresses this by providing a single, extensible format that can represent a wide variety of compression schemes.

* **Unified Checkpoint Format**: Supports various compression schemes in a single, consistent format.
* **Wide Compatibility**: Works with popular quantization methods like GPTQ, SmoothQuant, AWQ, AutoRound, etc. See [llm-compressor](https://github.com/vllm-project/llm-compressor)
* **Flexible Quantization Support**:
  * Activation quantization
  * Mixed precision
  * Low/arbitrary-bit
  * KV cache quantization
  * Non-uniform schemes (different layers can be quantized in different ways!)
* **Sparsity Support**: Handles both unstructured and semi-structured (e.g., 2:4) sparsity patterns.
* **Transform Support**: Rotation-based quantization techniques (Hadamard, random Hadamard, random matrix transforms).
* **Checkpoint Conversion**: Convert between formats like AutoAWQ, ModelOpt NVFP4, FP8 block, and compressed-tensors.
* **Model Offloading**: Transparent CPU/disk/distributed offloading for models larger than available VRAM.
* **Open-Source Integration**: Designed to work seamlessly with Hugging Face models, PyTorch, [vLLM](https://github.com/vllm-project/vllm), and [SGLang](https://github.com/sgl-project/sglang).

This allows developers and researchers to easily experiment with composing different quantization methods, simplify model deployment pipelines, and reduce the overhead of supporting multiple compression formats in inference engines.

## Next steps

* [Getting Started](getting-started/install.md) — install `compressed-tensors` and compress your first model
* [API Reference](api/index.md) — auto-generated reference for every public module
* Browse [`examples/`](https://github.com/vllm-project/compressed-tensors/tree/main/examples) on GitHub for calibrated quantization and checkpoint conversion walkthroughs
