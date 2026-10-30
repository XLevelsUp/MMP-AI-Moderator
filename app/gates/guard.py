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
    signals: set[str] = field(default_factory=set)           # act_from signals + review-band signals
    review: set[str] = field(default_factory=set)            # the review-band ones among them


def spread(count: int, wanted: int) -> list[int]:
    if wanted <= 1:
        return [0]
    return sorted({round(i * (count - 1) / (wanted - 1)) for i in range(wanted)})


def guard_frames(count: int, is_video: bool, suspicion=None) -> list[int]:
    """Which prepared frames are asked. An image: its one frame. A video: video_frames in all;
    the video_suspicious frames SigLIP finds most suspicious, each with the frame after it (a
    problem lasts more than one frame, see video_review_min_frames), and the rest spread from
    the first to the last frame. Without SigLIP scores: all of them spread."""
    if not is_video or count <= 1:
        return [0]
    wanted = min(settings.gate2.guard.video_frames, count)
    if suspicion is None:
        return spread(count, wanted)
    chosen = focus_frames(count, suspicion)
    if len(chosen) < wanted:   # the remaining places spread over the whole video, first to last frame
        chosen |= set(spread(count, wanted - len(chosen)))
    return sorted(chosen)


def focus_frames(count: int, suspicion) -> set[int]:
    """The video_suspicious frames SigLIP finds most suspicious, each with the frame after it.
    Only these get the pet crop as well: the crop is what lifted a bite moment into the review band,
    and asking it on every spread frame cost a quarter of the video's guard time."""
    cfg = settings.gate2.guard
    wanted = min(2 * cfg.video_suspicious, cfg.video_frames, count)
    chosen: set[int] = set()
    for i in sorted(range(count), key=lambda i: -float(suspicion[i])):
        if len(chosen) >= wanted:
            break
        chosen |= {i, min(i + 1, count - 1)}
    return chosen


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
    if min(right - left, bottom - top) < cfg.crop_min_side:
        return None   # too small to add anything (a ~28 px crop is one picture token and broke the model)
    if (right - left) * (bottom - top) > cfg.crop_max_area * width * height:
        return None
    return picture.crop((int(left), int(top), int(round(right)), int(round(bottom))))


def guard_views(picture: Image.Image, box) -> list[Image.Image]:
    crop = pet_crop(picture, box)
    return [fit(picture)] + ([fit(crop)] if crop is not None else [])


def guard_signals(scores: dict[str, float]) -> tuple[set[str], set[str]]:
    """Each question's signal from act_from, its review_signal from review_from: (act, review)."""
    act, review = set(), set()
    for name, question in settings.gate2.guard.questions.items():
        value = scores.get(name, 0.0)
        if value >= question.act_from:
            act.add(question.signal)
        elif value >= question.review_from:
            review.add(question.review_signal)
    return act, review


def steady_review(results: list[GuardResult]) -> list[GuardResult]:
    """Video: a review-band signal counts only where it shows in at least video_review_min_frames
    of the asked frames (one borderline frame out of many is noise; spec 6.3 also judges a video
    by runs of frames). Signals from act_from always count."""
    need = settings.gate2.guard.video_review_min_frames
    seen: dict[str, int] = {}
    for result in results:
        for signal in result.review:
            seen[signal] = seen.get(signal, 0) + 1
    for result in results:
        dropped = {s for s in result.review if seen[s] < need}
        result.review -= dropped
        result.signals -= dropped
    return results


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
        act, review = guard_signals(best)
        results.append(GuardResult(scores=best, signals=act | review, review=review))
    return results
