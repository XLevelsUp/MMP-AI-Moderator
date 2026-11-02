"""Gate 1 head, step 3: rules vs the trained head on the hard-case TEST set, same frames, same models.

Each photo goes through the real Gate 1 (prepare, SigLIP 2, OWLv2 + BioCLIP 2), then gets two
answers: the rules (fused score) and the head. Sub-folders say what is right:
    pet/     an animal is there and it may be approved      (wrong: rejected, or sent to review)
    no_pet/  must be rejected                               (wrong: approved)
    review/  printed, judged by hand
The head ships only if it lets fewer no-pet photos through without stopping more pets.

    uv run python -m scripts.head_eval D:/samples/hard_case_clean
    uv run python -m scripts.head_eval D:/samples/hard_case_clean --cached   (thresholds only, no models)
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from app.core.config import PROJECT_ROOT, settings
from app.gates.detection import detect_animals
from app.gates.gate1 import run_gate1
from app.prepare.images import prepare_image

CACHE = PROJECT_ROOT / "data" / "gate1_head" / "eval_cache.npz"
REPORT = PROJECT_ROOT / "data" / "gate1_head" / "eval.csv"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def answer(result) -> str:
    if result.outcome == "fail":
        return "REJECTED"
    if result.outcome == "borderline" or result.wild:
        return "REVIEW"
    return "APPROVED"


def wrong(expected: str, got: str) -> bool:
    return (expected == "pet" and got != "APPROVED") or (expected == "no_pet" and got == "APPROVED")


def run_models(root: Path) -> dict:
    """Real Gate 1 per photo: rules answer + the head's probabilities (kept for threshold sweeps)."""
    from app.ml.registry import get_backbone, get_detector, get_species_model, load_models
    from app.ml.head import Gate1Head

    for name in ("nsfw_model", "ai_model", "audio_model", "speech_model", "guard_model"):
        setattr(settings.ml, name, None)   # Gate 1 needs only SigLIP, OWLv2 and BioCLIP
    settings.ml.ocr = False
    load_models()
    backbone, detector, species = get_backbone(), get_detector(), get_species_model()
    head = Gate1Head()
    head.load()

    rows = {"files": [], "expected": [], "rules": [], "rules_score": [], "head": []}
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    for n, path in enumerate(files, 1):
        frame = prepare_image(path).frames[0]
        embeds = backbone.embed_images([frame.backbone])
        detections = detect_animals([frame.detector], detector, species)
        rules = run_gate1(backbone.similarities(embeds), backbone.labels, backbone.scale, False, detections)
        features = {"siglip": embeds, "bioclip": species.embed([frame.detector]).cpu().numpy()}
        probs = head.probabilities(features)[0]
        rows["files"].append(str(path.relative_to(root)))
        rows["expected"].append(path.relative_to(root).parts[0])
        rows["rules"].append(answer(rules))
        rows["rules_score"].append(rules.pet_score)
        rows["head"].append([probs[c] for c in ("pet", "wild", "no_pet")])
        print(f"{n}/{len(files)} {rows['expected'][-1]:>7} rules {rows['rules'][-1]:<8} "
              f"head pet {probs['pet']:.2f} wild {probs['wild']:.2f} no_pet {probs['no_pet']:.2f}  {path.name}",
              flush=True)
    return {k: np.array(v) for k, v in rows.items()}


def head_answer(probs, pass_from: float, fail_below: float, wild_from: float) -> str:
    animal = probs[0] + probs[1]
    if animal < fail_below:
        return "REJECTED"
    if animal < pass_from or probs[1] / animal >= wild_from:
        return "REVIEW"
    return "APPROVED"


def tally(expected, answers) -> dict:
    pet, no_pet = expected == "pet", expected == "no_pet"
    return {
        "no_pet approved": int((no_pet & (answers == "APPROVED")).sum()),
        "pet rejected": int((pet & (answers == "REJECTED")).sum()),
        "pet to review": int((pet & (answers == "REVIEW")).sum()),
        "no_pet to review": int((no_pet & (answers == "REVIEW")).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cached", action="store_true", help="reuse the last model run")
    args = parser.parse_args()

    if args.cached:
        data = dict(np.load(CACHE))
    else:
        data = run_models(args.root)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **data)
    expected, rules, probs = data["expected"], data["rules"], data["head"]
    judged = np.isin(expected, ["pet", "no_pet"])
    print(f"\n{int((expected == 'pet').sum())} pet, {int((expected == 'no_pet').sum())} no_pet judged")

    h = settings.gate1.head
    print("\nanswer                         no_pet approved  pet rejected  pet->review  no_pet->review")
    t = tally(expected, rules)
    print(f"rules today                    {t['no_pet approved']:>15} {t['pet rejected']:>13} {t['pet to review']:>12} {t['no_pet to review']:>15}")
    for pass_from in (0.4, 0.5, 0.6, 0.7, 0.8):
        for fail_below in (0.1, 0.2, 0.3):
            if fail_below > pass_from:
                continue
            answers = np.array([head_answer(p, pass_from, fail_below, h.wild_from) for p in probs])
            t = tally(expected, answers)
            mark = "  <- settings" if (pass_from, fail_below) == (h.pass_from, h.fail_below) else ""
            print(f"head pass {pass_from:.1f} fail {fail_below:.1f}         {t['no_pet approved']:>15} "
                  f"{t['pet rejected']:>13} {t['pet to review']:>12} {t['no_pet to review']:>15}{mark}")

    for bar in (0.6, 0.7, 0.8, 0.9):   # decider check: the rules decide, a pass the head calls no_pet -> review
        checked = np.where((rules == "APPROVED") & (probs[:, 2] >= bar), "REVIEW", rules)
        t = tally(expected, checked)
        mark = "  <- settings" if bar == h.check_no_pet_from else ""
        print(f"check no_pet from {bar:.1f}         {t['no_pet approved']:>15} "
              f"{t['pet rejected']:>13} {t['pet to review']:>12} {t['no_pet to review']:>15}{mark}")

    answers = np.array([head_answer(p, h.pass_from, h.fail_below, h.wild_from) for p in probs])
    print(f"\nwith the settings (pass {h.pass_from}, fail {h.fail_below}, wild {h.wild_from}), files that changed or are wrong:")
    with REPORT.open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh)
        out.writerow(["file", "expected", "rules", "head", "pet", "wild", "no_pet"])
        for i, f in enumerate(data["files"]):
            out.writerow([f, expected[i], rules[i], answers[i], *(round(float(v), 3) for v in probs[i])])
            if not judged[i] or rules[i] != answers[i] or wrong(expected[i], answers[i]):
                flag = "WRONG" if judged[i] and wrong(expected[i], answers[i]) else ("fixed" if judged[i] and wrong(expected[i], rules[i]) else "")
                print(f"  {flag:5} {expected[i]:>7} rules {rules[i]:<8} head {answers[i]:<8} "
                      f"pet {probs[i][0]:.2f} wild {probs[i][1]:.2f} no_pet {probs[i][2]:.2f}  {f}")
    print(f"\nper file: {REPORT}")


if __name__ == "__main__":
    main()
