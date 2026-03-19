# compressed-tensors

The `compressed-tensors` library extends the [safetensors](https://github.com/huggingface/safetensors) format, providing a versatile and efficient way to store and manage compressed tensor data. This library supports various quantization and sparsity schemes, making it a unified format for handling different model optimizations like GPTQ, AWQ, SmoothQuant, INT8, FP8, SparseGPT, and more.

## Why `compressed-tensors`?

As model compression becomes increasingly important for efficient deployment of LLMs, the landscape of quantization and compression techniques has become increasingly fragmented.
Each method often comes with its own storage format and loading procedures, making it challenging to work with multiple techniques or switch between them.
`compressed-tensors` addresses this by providing a single, extensible format that can represent a wide variety of compression schemes. 

* **Unified Checkpoint Format**: Supports various compression schemes in a single, consistent format.
* **Wide Compatibility**: Works with popular quantization methods like GPTQ, SmoothQuant, and FP8. See [llm-compressor](https://github.com/vllm-project/llm-compressor)
* **Flexible Quantization Support**: 
  * Weight-only quantization (e.g., W4A16, W8A16, WnA16)
  * Activation quantization (e.g., W8A8)
  * KV cache quantization
  * Non-uniform schemes (different layers can be quantized in different ways!)
* **Sparsity Support**: Handles both unstructured and semi-structured (e.g., 2:4) sparsity patterns.
* **Open-Source Integration**: Designed to work seamlessly with Hugging Face models and PyTorch.

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

## Getting started

## Saving a Compressed Model with PTQ

We can use compressed-tensors to run basic post training quantization (PTQ) and save the quantized model compressed on disk

```python
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", torch_dtype="auto")

config = QuantizationConfig.parse_file("./examples/bit_packing/int4_config.json")
config.quantization_status = QuantizationStatus.CALIBRATION
apply_quantization_config(model, config)

dataset = load_dataset("ptb_text_only")["train"]
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding=False, truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_loader = DataLoader(tokenized_dataset, batch_size=1, collate_fn=DefaultDataCollator())

with torch.no_grad():
    for idx, sample in tqdm(enumerate(data_loader), desc="Running calibration"):
        sample = {key: value.to(device) for key,value in sample.items()}
        _ = model(**sample)

        if idx >= 512:
            break

model.apply(freeze_module_quantization)
model.apply(compress_quantized_weights)

output_dir = "./ex_llama1.1b_w4a16_packed_quantize"
compressor = ModelCompressor.from_pretrained_model(model)
compressor.compress_model(model)
model.save_pretrained(output_dir)
```
