import base64
import io
import logging
import math

import app.core.config  # noqa: F401  loads .env (HF_HOME, offline) before transformers
import torch
from PIL import Image

from app.core.config import settings


logger = logging.getLogger("mmp.ml")

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruction provided. Note that the answer can only be "yes" or "no".'
)
YES = ("yes", "yes.", '"yes"', "'yes'")
NO = ("no", "no.", '"no"', "'no'")


class GuardModel:
    """Shieldstral 1.0: one yes / no policy question about one picture -> probability of "yes".

    Prompt format and scoring follow the model card (mistralai/Shieldstral-1.0-3B): the
    "yes" / "no" logits at the answer position, renormalised over those two words.
    """

    def __init__(self, name: str):
        self.name = name
        self.spec = settings.ml.models[name]
        self.tokenizer = None
        self.model = None
        self.device = torch.device(settings.ml.device)
        # bfloat16 where the GPU has it (L4); float16 on a T4; bfloat16 on CPU (float32 needs ~16 GB)
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        else:
            self.dtype = torch.bfloat16

    def load(self) -> None:
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        self.tokenizer = MistralCommonBackend.from_pretrained(self.spec.repo, revision=self.spec.revision)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.spec.repo, revision=self.spec.revision, dtype=self.dtype
        ).eval().to(self.device)
        logger.info("%s loaded (%s @ %s, %s)", self.name, self.spec.repo, self.spec.revision[:7], self.dtype)

    def warmup(self) -> None:
        self.score(Image.new("RGB", (64, 64), "gray"), "Is there an animal?", settings.gate2.guard.instruct)

    @staticmethod
    def data_uri(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    @torch.inference_mode()
    def score(self, image: Image.Image, question: str, instruct: str) -> float:
        """Probability that the answer to `question` about `image` is "yes"."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"<Instruct>: {instruct}\n\n<Query>: {question}\n\n<Document>: "},
                {"type": "image_url", "image_url": {"url": self.data_uri(image)}},
                {"type": "text", "text": "\n\n"},
            ]},
        ]
        encoded = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
        inputs = {
            key: (value.to(self.device, self.dtype) if key == "pixel_values" else value.to(self.device))
            if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }
        logprobs = torch.log_softmax(self.model(**inputs).logits[0, -1].float(), dim=-1)
        z_yes = z_no = -1e9
        values, ids = logprobs.topk(20)
        for logp, token_id in zip(values.tolist(), ids.tolist()):
            word = self.tokenizer.decode([token_id]).strip().lower()
            if word in YES:
                z_yes = max(z_yes, logp)
            elif word in NO:
                z_no = max(z_no, logp)
        return math.exp(z_yes) / (math.exp(z_yes) + math.exp(z_no))

    def scores(self, images: list[Image.Image], questions: dict[str, str], instruct: str) -> list[dict[str, float]]:
        """Every question for every picture: one {question name: probability} per picture."""
        return [{name: self.score(image, text, instruct) for name, text in questions.items()} for image in images]
