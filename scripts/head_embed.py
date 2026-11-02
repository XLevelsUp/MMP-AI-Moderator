"""Gate 1 head, step 1: turn every labelled photo into the numbers the pipeline already makes.

For each row of labels.csv (file, class, kind, ...) the photo is prepared exactly as an upload
(app.prepare.images), then
    SigLIP 2  -> 768 numbers from the backbone picture (what Gate 1 already computes)
    BioCLIP 2 -> 768 numbers from the whole detector picture
Training photos also get one spoiled copy (blur / dark / tiny / glare / heavy JPEG), so the head
learns that a bad phone photo of a dog is still a dog. Exact duplicate files are kept once.

    uv run python -m scripts.head_embed D:/dataset            -> data/gate1_head/embeddings.npz
    uv run python -m scripts.head_embed D:/dataset --copies 0  (no spoiled copies)
    ML_DEVICE=cuda ML_DTYPE=float16 python -m scripts.head_embed /content/dataset   (Colab, ~5 min on a T4)

Never point this at D:/samples/hard_case_clean: that is the test set.
"""
import argparse
import csv
import hashlib
import io
import random
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from app.core.config import PROJECT_ROOT, settings
from app.ml.bioclip import BioclipClassifier
from app.ml.runtime import configure_torch
from app.ml.siglip import SiglipEmbedder
from app.prepare.images import frame_from, to_rgb

CLASSES = ["pet", "wild", "no_pet"]
OUT = PROJECT_ROOT / "data" / "gate1_head" / "embeddings.npz"


def blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 3.5)))


def dark(img, rng):
    return ImageEnhance.Brightness(img).enhance(rng.uniform(0.25, 0.5))


def tiny(img, rng):
    side = rng.randint(64, 128)
    small = ImageOps.contain(img, (side, side))
    return small.resize(img.size, Image.Resampling.BILINEAR)


def glare(img, rng):
    w, h = img.size
    light = Image.new("L", img.size, 0)
    r = int(min(w, h) * rng.uniform(0.3, 0.6))
    x, y = rng.randint(0, w), rng.randint(0, h)
    ImageDraw.Draw(light).ellipse((x - r, y - r, x + r, y + r), fill=230)
    light = light.filter(ImageFilter.GaussianBlur(r / 2))
    return Image.composite(Image.new("RGB", img.size, "white"), img, light)


def jpeg(img, rng):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(8, 20))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


SPOILERS = {f.__name__: f for f in (blur, dark, tiny, glare, jpeg)}


def load(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img.draft("RGB", (settings.prepare.detector_size,) * 2)
        return to_rgb(ImageOps.exif_transpose(img))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--labels", type=Path, help="default: <root>/labels.csv")
    parser.add_argument("--copies", type=int, default=1, help="spoiled copies per photo")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, help="first N rows only (a quick dry run)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if "hard_case" in str(args.root):
        raise SystemExit("refusing: the hard-case set is the test set")

    configure_torch()
    siglip = SiglipEmbedder(settings.ml.backbone, prompts={"photo": "a photo"})
    bioclip = BioclipClassifier(settings.ml.species_model, species=["animal"])
    siglip.load()
    bioclip.load()

    rows = list(csv.DictReader((args.labels or args.root / "labels.csv").open(encoding="utf-8-sig")))[: args.limit]
    rng = random.Random(args.seed)
    seen: set[str] = set()
    out = {k: [] for k in ("siglip", "bioclip", "label", "group", "spoiler", "files", "kind")}
    started = perf_counter()
    for n, row in enumerate(rows):
        path = args.root / row["file"].replace("\\", "/")   # labels.csv is written on Windows
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        original = load(path)
        versions = [("none", original)]
        for _ in range(args.copies):
            name = rng.choice(list(SPOILERS))
            versions.append((name, SPOILERS[name](original, rng)))
        frames = [frame_from(img) for _, img in versions]
        out["siglip"].append(siglip.embed_images([f.backbone for f in frames]))
        out["bioclip"].append(bioclip.embed([f.detector for f in frames]).cpu().numpy())
        for name, _ in versions:
            out["label"].append(CLASSES.index(row["class"]))
            out["group"].append(len(seen) - 1)
            out["spoiler"].append(name)
            out["files"].append(row["file"])
            out["kind"].append(row["kind"])
        if (n + 1) % 50 == 0:
            done = perf_counter() - started
            print(f"{n + 1}/{len(rows)}  {done / (n + 1):.2f} s per photo  ~{done / (n + 1) * (len(rows) - n - 1) / 60:.0f} min left",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        siglip=np.concatenate(out["siglip"]).astype(np.float32),
        bioclip=np.concatenate(out["bioclip"]).astype(np.float32),
        **{k: np.array(out[k]) for k in ("label", "group", "spoiler", "files", "kind")},
        classes=np.array(CLASSES),
        siglip_model=np.array(f"{settings.ml.backbone}@{siglip.spec.revision}"),
        bioclip_model=np.array(f"{settings.ml.species_model}@{bioclip.spec.revision}"),
    )
    print(f"{len(seen)} photos ({len(rows) - len(seen)} duplicates skipped), {len(out['label'])} rows -> {args.out}")


if __name__ == "__main__":
    main()
