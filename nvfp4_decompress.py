from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils import dispatch_for_generation

#MODEL_ID = "nm-testing/TinyLlama-1.1B-Chat-v1.0-w4a16-asym-awq-e2e"
MODEL_ID = "nm-testing/TinyLlama-1.1B-Chat-v1.0-NVFP4"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))''
print("==========================================\n\n")
