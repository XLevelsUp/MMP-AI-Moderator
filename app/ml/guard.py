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
        # Straight onto the GPU: loading the ~8 GB to RAM first was killed on a Colab T4 (12.7 GB RAM)
        # with the other models already loaded.
        on_gpu = {"device_map": str(self.device)} if self.device.type == "cuda" else {}
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.spec.repo, revision=self.spec.revision, dtype=self.dtype, **on_gpu
        ).eval()
        if not on_gpu:
            self.model.to(self.device)
        logger.info("%s loaded (%s @ %s, %s)", self.name, self.spec.repo, self.spec.revision[:7], self.dtype)

    def warmup(self) -> None:
        self.scores([Image.new("RGB", (64, 64), "gray")], {"warmup": "Is there an animal?"},
                    settings.gate2.guard.instruct)

    @staticmethod
    def data_uri(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def conversation(uri: str, question: str, instruct: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"<Instruct>: {instruct}\n\n<Query>: {question}\n\n<Document>: "},
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "\n\n"},
            ]},
        ]

    def yes_probability(self, logits: torch.Tensor) -> float:
        """"yes" vs "no" at the answer position (model card scoring)."""
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        z_yes = z_no = -1e9
        values, ids = logprobs.topk(20)
        for logp, token_id in zip(values.tolist(), ids.tolist()):
            word = self.tokenizer.decode([token_id]).strip().lower()
            if word in YES:
                z_yes = max(z_yes, logp)
            elif word in NO:
                z_no = max(z_no, logp)
        return math.exp(z_yes) / (math.exp(z_yes) + math.exp(z_no))

    @torch.inference_mode()
    def ask(self, image: Image.Image, questions: list[str], instruct: str) -> list[float]:
        """Several questions about one picture in GPU batches of gate2.guard.batch_size.

        The vision encoder reads the picture once and every question in the batch reuses it:
        giving the model the picture once per row ran a T4 out of memory (the encoder joins all
        pictures of a batch into one sequence, so 11 copies cost 121x the attention memory).
        The tokenizer pads on the left, so the last position of every row is its answer slot;
        only that position's logits are computed."""
        uri = self.data_uri(image)
        size = settings.gate2.guard.batch_size
        picture = None   # the vision encoder's output for this picture
        out = []
        for start in range(0, len(questions), size):
            chunk = [self.conversation(uri, q, instruct) for q in questions[start:start + size]]
            encoded = self.tokenizer.apply_chat_template(chunk, return_tensors="pt", return_dict=True, padding=True)
            if picture is None:
                picture = self.model.model.get_image_features(
                    pixel_values=encoded["pixel_values"][:1].to(self.device, self.dtype),
                    image_sizes=encoded["image_sizes"][:1].to(self.device),
                    vision_feature_layer=self.model.config.vision_feature_layer,
                    return_dict=True,
                )
                features = picture.pooler_output[0]
            picture.pooler_output = (features,) * len(chunk)
            last = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
                mm_encoder_outputs={"image": picture},
                logits_to_keep=1,
            ).logits[:, -1]
            out += [self.yes_probability(row) for row in last]
        return out

    def scores(self, images: list[Image.Image], questions: dict[str, str], instruct: str) -> list[dict[str, float]]:
        """Every question for every picture: one {question name: probability} per picture."""
        names = list(questions)
        return [dict(zip(names, self.ask(image, [questions[n] for n in names], instruct))) for image in images]
