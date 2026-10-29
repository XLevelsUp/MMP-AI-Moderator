"""Re-check the guard thresholds on SAFE photos (run after changing gate2.guard questions, wording,
image_size or the model). Uses the pipeline's own guard code: full frame + the pet crop from the
detector (OWLv2), every question, the current thresholds.

    GUARD_MODEL=shieldstral ML_DEVICE=cuda ML_DTYPE=float16 python -m scripts.guard_calibrate ../data \\
        --review-ok safety_check/play_fighting

Every photo under the folder is treated as safe; folders after --review-ok are ones where a review is
intended (e.g. play fighting) and are left out of the threshold suggestion. Prints, per question, the
safe-photo p95 / p99 / max and the review_from that keeps at most --max-review of normal photos below
it, and how many photos the CURRENT thresholds send to review per folder. Writes guard_scores.csv.
"""
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import settings
from app.gates.detection import detect_animals
from app.gates.guard import guard_signals, run_guard
from app.ml.registry import get_detector, get_guard_model, load_models
from app.prepare.images import fit

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--review-ok", nargs="*", default=[], help="sub-folders where review is intended")
    parser.add_argument("--max-review", type=float, default=0.01, help="share of normal photos allowed to review")
    args = parser.parse_args()

    for role in ("backbone", "species_model", "nsfw_model", "ai_model", "audio_model", "speech_model"):
        setattr(settings.ml, role, None)   # only the detector (for the crop) and the guard
    settings.ml.ocr = False
    load_models()
    guard, detector = get_guard_model(), get_detector()
    if guard is None:
        raise SystemExit("guard model not loaded: set GUARD_MODEL=shieldstral (and a GPU)")

    photos = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    cfg = settings.gate2.guard
    print(f"{len(photos)} photos, {len(cfg.questions)} questions, image_size {cfg.image_size}", flush=True)
    rows, per_folder = [], defaultdict(Counter)
    for n, path in enumerate(photos, 1):
        folder = path.parent.relative_to(args.folder).as_posix()
        with Image.open(path) as img:
            picture = fit(img.convert("RGB"), settings.prepare.detector_size)
        box = detect_animals([picture], detector)[0].box if detector is not None else None
        [result] = run_guard(guard, [picture], [box])
        act, review = guard_signals(result.scores)
        verdict = "act" if act else "review" if review else "ok"
        per_folder[folder][verdict] += 1
        if verdict != "ok":
            top = sorted(result.scores.items(), key=lambda kv: -kv[1])[:3]
            print(f"  {verdict:6s} {folder}/{path.name}: " + ", ".join(f"{k} {v:.2f}" for k, v in top), flush=True)
        rows.append({"folder": folder, "file": path.name, **{k: round(v, 3) for k, v in result.scores.items()}})
        if n % 25 == 0:
            print(f"{n}/{len(photos)}", flush=True)

    with open("guard_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nCurrent thresholds, per folder (ok / review / act):")
    for folder in sorted(per_folder):
        c = per_folder[folder]
        print(f"  {folder:40s} {c['ok']:3d} / {c['review']:3d} / {c['act']:3d}")

    normal = [r for r in rows if not any(r["folder"].startswith(f) for f in args.review_ok)]
    print(f"\nSafe-photo scores ({len(normal)} normal photos) and suggested review_from "
          f"(at most {args.max_review:.0%} of them above it):")
    for name, question in cfg.questions.items():
        values = np.array(sorted(r[name] for r in normal))
        allowed = int(math.floor(len(values) * args.max_review))
        edge = values[-(allowed + 1)] if len(values) > allowed else values[-1]
        suggest = max(0.15, math.ceil((edge + 0.01) * 20) / 20)   # next 0.05 step above that score
        print(f"  {name:12s} p95 {np.percentile(values, 95):.3f}  p99 {np.percentile(values, 99):.3f}  "
              f"max {values[-1]:.3f}   now {question.review_from:.2f} -> suggest {min(suggest, question.act_from - 0.05):.2f}")
    print("Saved guard_scores.csv")


if __name__ == "__main__":
    main()
