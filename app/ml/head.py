import hashlib
import logging

import numpy as np

from app.core.config import PROJECT_ROOT, settings


logger = logging.getLogger("mmp.ml")


class Gate1Head:
    """The trained Gate 1 head (scripts.head_train): pet / wild / no_pet from the SigLIP 2
    and BioCLIP 2 embeddings the pipeline already makes. Plain numbers, so numpy is enough."""

    def __init__(self, name: str = "gate1_head"):
        self.name = name
        self.path = PROJECT_ROOT / settings.gate1.head.file
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.classes: list[str] = []
        self.features: list[str] = []
        self.digest = ""

    def load(self) -> None:
        data = np.load(self.path)
        ml = settings.ml
        # Embeddings from another model revision mean different numbers: refuse, never guess.
        expected = {"siglip": f"{ml.backbone}@{ml.models[ml.backbone].revision}"}
        if ml.species_model:
            expected["bioclip"] = f"{ml.species_model}@{ml.models[ml.species_model].revision}"
        self.features = [str(f) for f in data["features"]]
        for feature in self.features:
            trained = str(data[f"{feature}_model"])
            if expected.get(feature) != trained:
                raise RuntimeError(f"head trained on {trained}, server runs {expected.get(feature)}")
        self.weights, self.bias = data["weights"], data["bias"]
        self.classes = [str(c) for c in data["classes"]]
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()[:8]
        logger.info("%s loaded (%s, %s, %s)", self.name, self.path.name, "+".join(self.features),
                    str(data["trained_on"]))

    def warmup(self) -> None:
        pass

    def probabilities(self, features: dict[str, np.ndarray]) -> list[dict[str, float]]:
        """{class: probability} per picture (softmax of the linear scores)."""
        x = np.concatenate([features[f] for f in self.features], axis=1)
        logits = x @ self.weights.T + self.bias
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return [dict(zip(self.classes, map(float, row))) for row in p]
