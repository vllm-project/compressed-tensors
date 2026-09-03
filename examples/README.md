# Compressing with compressed-tensors

The examples in this directory show how to compress models using compressed-tensors primitives directly. This includes applying quantization configs, computing scales via PyTorch hooks or manual calculation, and saving models in the compressed-tensors format. These examples use basic round-to-nearest quantization without advanced calibration algorithms.

# Compressing with LLM Compressor

Naively quantizing your models doesn't always maintain model accuracy, especially at low bit widths. To preserve accuracy, you often need to calibrate the quantization parameters to minimize calibration error. This can be accomplished by applying algorithms such as GPTQ, AWQ, AutoRound, etc. LLM Compressor provides support for these algorithms, leveraging the compressed-tensors primitives under the hood without requiring the user to interact with them directly. The final model is saved in the compressed-tensors format, enabling seamless inference with vLLM.

For a step-by-step guide on compressing your model (including selecting your quantization algorithm for optimized quantization parameters that maintain accuracy at low bit widths), check out: https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-model/

For a list of comprehensive LLM Compressor examples, check out: https://github.com/vllm-project/llm-compressor/tree/main/examples

