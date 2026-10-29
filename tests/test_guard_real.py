"""GuardModel.ask (picture read once, questions batched) gives the same answers as asking each
question on its own. Uses the real Shieldstral tokenizer and config, but a tiny random-weight
model, so it runs on a CPU in seconds (only the small tokenizer files must be downloaded)."""
import pytest
import torch
from PIL import Image

from app.core.config import settings
from app.ml.guard import GuardModel

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def tiny_guard():
    from transformers import AutoConfig, Mistral3ForConditionalGeneration, MistralCommonBackend

    spec = settings.ml.models["shieldstral"]
    config = AutoConfig.from_pretrained(spec.repo, revision=spec.revision)
    for part in (config.text_config, config.vision_config):
        part.num_hidden_layers, part.hidden_size, part.intermediate_size = 1, 64, 128
        part.num_attention_heads, part.head_dim = 2, 32
    config.text_config.num_key_value_heads = 1
    torch.manual_seed(0)
    guard = GuardModel.__new__(GuardModel)
    guard.tokenizer = MistralCommonBackend.from_pretrained(spec.repo, revision=spec.revision)
    guard.model = Mistral3ForConditionalGeneration(config).eval()
    guard.device, guard.dtype = torch.device("cpu"), torch.float32
    guard.yes_probability = lambda row: row.clone()   # compare the raw answer-slot logits
    return guard


@pytest.mark.parametrize("batch_size", [11, 4])
def test_batched_questions_match_one_at_a_time(tiny_guard, monkeypatch, batch_size):
    monkeypatch.setattr(settings.gate2.guard, "batch_size", batch_size)
    instruct = settings.gate2.guard.instruct
    questions = [q.text for q in settings.gate2.guard.questions.values()]
    picture = Image.new("RGB", (320, 240), "orange")

    batched = tiny_guard.ask(picture, questions, instruct)

    uri = tiny_guard.data_uri(picture)
    with torch.inference_mode():
        for question, got in zip(questions, batched):
            encoded = tiny_guard.tokenizer.apply_chat_template(
                tiny_guard.conversation(uri, question, instruct), return_tensors="pt", return_dict=True)
            expected = tiny_guard.model(**encoded).logits[0, -1]
            assert torch.allclose(expected, got, atol=1e-4)
