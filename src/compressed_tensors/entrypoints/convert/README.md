# Converting Checkpoints Without Loading Models

`convert_checkpoint` provides a pathway to convert quantized model checkpoints from other formats to compressed-tensors format without loading the model into memory. This entrypoint operates directly on safetensors files in the checkpoint, converting quantization parameters and updating the quantization_config in config.json.

## Use Cases

This pathway is useful when you need to convert checkpoints that are already quantized in another format (e.g., ModelOpt NVFP4, AutoAWQ) and need to be converted to compressed-tensors format before they can be further compressed.

## How It Works

`convert_checkpoint` processes checkpoints by:

1. Iterating over safetensors files in the checkpoint
2. Applying `Converter` instances to transform tensor names and values
3. Saving the converted tensors to a new checkpoint
4. Updating config.json with the appropriate compressed-tensors quantization_config

## Converter System

The conversion logic is implemented through the `Converter` protocol, which defines:

- `validate(tensors)`: Validate that tensors can be converted
- `process(tensors)`: Transform tensor names and values in-place
- `create_config()`: Generate the QuantizationConfig

### Available Converters

**ModelOptNvfp4Converter**: Converts NVIDIA ModelOpt NVFP4 checkpoints to compressed-tensors NVFP4 format
- Transforms parameter names (e.g., `weight` → `weight_packed`, `weight_scale_2` → `weight_global_scale`)
- Inverts scale tensors to match compressed-tensors conventions
- Supports targeted conversion with `ignore` and `targets` patterns

## Usage Example

Examples available at `examples/convert_checkpoint`.