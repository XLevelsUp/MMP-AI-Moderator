"""D1-D4 from yes / no policy questions (settings.gate2.guard, ml.guard_model = Shieldstral).

Each chosen frame is asked every question twice: on the whole picture and on a crop around
the detector's pet box, grown to include whoever is handling the animal. The higher of the
two scores counts. Score >= act_from fires the question's signal; review_from..act_from fires
its review_signal, so a human looks.
"""
from dataclasses import dataclass, field

from PIL import Image

from app.core.config import settings


@dataclass
class GuardResult:
    scores: dict[str, float] = field(default_factory=dict)   # question -> best score over the views
    signals: set[str] = field(default_factory=set)


def guard_frames(count: int, is_video: bool) -> list[int]:
    """Which prepared frames are asked: the one image, or gate2.guard.video_frames of a video
    spread from the first to the last frame."""
    if not is_video or count <= 1:
        return [0]
    wanted = min(settings.gate2.guard.video_frames, count)
    if wanted == 1:
        return [0]
    return sorted({round(i * (count - 1) / (wanted - 1)) for i in range(wanted)})


def fit(picture: Image.Image) -> Image.Image:
    size = settings.gate2.guard.image_size
    if max(picture.size) <= size:
        return picture
    copy = picture.copy()
    copy.thumbnail((size, size))
    return copy


def pet_crop(picture: Image.Image, box: tuple[float, float, float, float] | None) -> Image.Image | None:
    """The pet box grown by crop_margin on each side (to take in the person handling the animal),
    or None when there is no box or the grown box covers most of the picture anyway."""
    if box is None:
        return None
    cfg = settings.gate2.guard
    width, height = picture.size
    x0, y0, x1, y1 = box
    grow_x, grow_y = (x1 - x0) * cfg.crop_margin, (y1 - y0) * cfg.crop_margin
    left, top = max(0.0, x0 - grow_x), max(0.0, y0 - grow_y)
    right, bottom = min(float(width), x1 + grow_x), min(float(height), y1 + grow_y)
    if right - left < 1 or bottom - top < 1:
        return None
    if (right - left) * (bottom - top) > cfg.crop_max_area * width * height:
        return None
    return picture.crop((int(left), int(top), int(round(right)), int(round(bottom))))


def guard_views(picture: Image.Image, box) -> list[Image.Image]:
    crop = pet_crop(picture, box)
    return [fit(picture)] + ([fit(crop)] if crop is not None else [])


def guard_signals(scores: dict[str, float]) -> set[str]:
    """Each question's signal from act_from, its review_signal from review_from."""
    found = set()
    for name, question in settings.gate2.guard.questions.items():
        value = scores.get(name, 0.0)
        if value >= question.act_from:
            found.add(question.signal)
        elif value >= question.review_from:
            found.add(question.review_signal)
    return found


def run_guard(model, pictures: list[Image.Image], boxes: list) -> list[GuardResult]:
    """One GuardResult per picture: the best score per question over its views, and the signals."""
    cfg = settings.gate2.guard
    questions = {name: q.text for name, q in cfg.questions.items()}
    results = []
    for picture, box in zip(pictures, boxes):
        best: dict[str, float] = {}
        for view_scores in model.scores(guard_views(picture, box), questions, cfg.instruct):
            for name, value in view_scores.items():
                best[name] = max(value, best.get(name, 0.0))
        results.append(GuardResult(scores=best, signals=guard_signals(best)))
    return results
