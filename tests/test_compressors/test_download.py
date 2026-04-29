import pytest


@pytest.mark.parametrize(
    "model_id", ["RedHatAI/gemma-3-12b-it-FP8-dynamic", "RedHatAI/Qwen3-8B-FP8-dynamic"]
)
def test_download_model(model_id):
    print("DOWNLOADING MODEL ", model_id)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("GOT MODEL ", model)
