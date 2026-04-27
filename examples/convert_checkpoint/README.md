# Convert Checkpoint Examples

This directory contains examples demonstrating how to use the `convert_checkpoint` entrypoint to convert model checkpoints between different quantization formats.

## Examples

### Single Converter Examples

**`qwen3_nvfp4_example.py`**
- Converts nvidia/Qwen3-32B-NVFP4 from ModelOpt NVFP4 format to compressed-tensors format
- Demonstrates `ModelOptNvfp4Converter` with KV cache quantization

**`qwen3_fpblock_example.py`**
- Dequantizes qwen-community/Qwen3-4B-FP8 from FP8 block quantization to dense bfloat16
- Demonstrates `FP8BlockDequantizer`

**`deepseek32_fpblock_example.py`**
- Dequantizes DeepSeek FP8 model to dense format
- Another example of `FP8BlockDequantizer`

### Multi-Converter Examples

**`simple_multi_converter.py`** ⭐ **Start here**
- Minimal, runnable example with Qwen2.5-0.5B (very small model)
- Shows how to pass multiple converters as a list
- Demonstrates custom converters that print info and scale weights
- Perfect for understanding the multi-converter workflow

**`multi_converter_example.py`**
- Comprehensive example with multiple patterns:
  - Custom `ScaleBiasConverter` for scaling weights
  - Custom `DtypeConverter` for changing tensor dtypes  
  - Practical example combining `FP8BlockDequantizer` with custom transformations
- Shows how converters are applied sequentially

## Usage

### Single Converter

```python
from compressed_tensors.entrypoints.convert import convert_checkpoint, ModelOptNvfp4Converter

convert_checkpoint(
    model_stub="nvidia/Qwen3-32B-NVFP4",
    save_directory="Qwen3-32B-CT",
    converter=ModelOptNvfp4Converter(
        targets=["re:.*mlp.*", "re:.*self_attn.*"],
    ),
    max_workers=8,
)
```

### Multiple Converters (NEW)

```python
from compressed_tensors.entrypoints.convert import convert_checkpoint

convert_checkpoint(
    model_stub="Qwen/Qwen2.5-0.5B-Instruct",
    save_directory="output",
    converter=[
        FirstConverter(...),   # Applied first
        SecondConverter(...),  # Applied second to results of first
        ThirdConverter(...),   # Applied third to results of second
    ],
    max_workers=4,
)
```

Converters are applied **sequentially** - each converter processes the tensors produced by the previous converter.

## Creating Custom Converters

A converter must implement the `Converter` protocol with these methods:

```python
from compressed_tensors.entrypoints.convert import Converter
import torch

class MyConverter(Converter):
    def process(self, tensors: dict[str, torch.Tensor]):
        """Transform the tensors (in-place or replace)."""
        pass
    
    def validate(self, tensors: dict[str, torch.Tensor]):
        """Validate tensors before processing."""
        pass
    
    def get_dependencies(self, weight_name: str) -> set[str]:
        """Return additional tensor names needed to process weight_name."""
        return set()
    
    def create_config(self, config: dict[str, object]) -> dict[str, object]:
        """Update quantization config based on conversion."""
        return config
```

See `simple_multi_converter.py` for complete examples.

## Running Examples

```bash
# Run the simple multi-converter example
python examples/convert_checkpoint/simple_multi_converter.py

# Run other examples (uncomment the desired function first)
python examples/convert_checkpoint/multi_converter_example.py
```

## Notes

- Use `max_workers` to control parallelism across shards
- Converters process each shard independently
- For multi-converter setups, all converters process the same shard before moving to the next
- Quantization configs are updated by calling each converter's `create_config()` in sequence
