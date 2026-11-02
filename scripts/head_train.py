"""Gate 1 head, step 2: learn pet / wild / no_pet from the embeddings (logistic regression).

20% of the photos (never their spoiled copies) are held back to check the head; then the head
is refit on everything and saved as plain numbers (weights + bias), so the API needs only numpy.

    uv run python -m scripts.head_train                      -> models/gate1_head.npz
    uv run python -m scripts.head_train --features siglip    (SigLIP only: no extra model call)

The report shows, on the held-back photos, what a pass bar on "animal" (pet + wild) would do:
how many no-pet photos it lets through and how many real animals it stops.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from app.core.config import PROJECT_ROOT

DATA = PROJECT_ROOT / "data" / "gate1_head" / "embeddings.npz"
OUT = PROJECT_ROOT / "models" / "gate1_head.npz"
FEATURES = {"siglip": ["siglip"], "siglip+bioclip": ["siglip", "bioclip"]}
STRENGTHS = [0.3, 1.0, 3.0, 10.0, 30.0]


def features(data, names: list[str]) -> np.ndarray:
    return np.concatenate([data[n] for n in names], axis=1)


def fit(x, y, c: float) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(x, y)


def animal_share(model, x) -> np.ndarray:
    """pet + wild probability: an animal is there (wild ones then go to review)."""
    p = model.predict_proba(x)
    return p[:, 0] + p[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--features", choices=list(FEATURES), default="siglip+bioclip")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = np.load(args.data)
    classes = [str(c) for c in data["classes"]]
    x, y, group = features(data, FEATURES[args.features]), data["label"], data["group"]
    original = data["spoiler"] == "none"
    kinds, files = data["kind"], data["files"]

    # One fixed 80/20 split by photo: a photo and its spoiled copy never sit on both sides.
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train, test = next(split.split(x, y, group))
    test_orig = test[original[test]]
    test_spoiled = test[~original[test]]

    print(f"features {args.features} ({x.shape[1]} numbers), train {len(train)} rows, "
          f"test {len(test_orig)} photos + {len(test_spoiled)} spoiled copies")
    best_c, best_acc = None, -1.0
    for c in STRENGTHS:
        model = fit(x[train], y[train], c)
        acc = (model.predict(x[test_orig]) == y[test_orig]).mean()
        print(f"  C={c:<5} accuracy {acc:.3f}  spoiled {(model.predict(x[test_spoiled]) == y[test_spoiled]).mean():.3f}")
        if acc > best_acc:
            best_c, best_acc = c, acc
    model = fit(x[train], y[train], best_c)

    predicted = model.predict(x[test_orig])
    truth = y[test_orig]
    print(f"\nC={best_c}: confusion on held-back photos (rows = truth, columns = head)")
    print("            " + "".join(f"{c:>8}" for c in classes))
    for i, name in enumerate(classes):
        print(f"  {name:10}" + "".join(f"{int(((truth == i) & (predicted == j)).sum()):>8}" for j in range(3)))

    animal = animal_share(model, x[test_orig])
    is_animal = truth != classes.index("no_pet")
    print("\npass bar on animal share -> no-pet photos let through / real animals stopped")
    for bar in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        through = int((~is_animal & (animal >= bar)).sum())
        stopped = int((is_animal & (animal < bar)).sum())
        print(f"  {bar:.1f}: {through}/{int((~is_animal).sum())} through, {stopped}/{int(is_animal.sum())} stopped")

    wrong = Counter()
    print("\nwrong on held-back photos:")
    for i in np.flatnonzero(predicted != truth):
        row = test_orig[i]
        wrong[str(kinds[row])] += 1
        print(f"  {classes[truth[i]]:>6} -> {classes[predicted[i]]:<6} {files[row]}")
    print("by kind:", dict(wrong.most_common()))

    print("\nheld-back accuracy by kind (photos / spoiled copies):")
    for kind in sorted(set(kinds[test_orig].tolist())):
        orig = test_orig[kinds[test_orig] == kind]
        spoiled = test_spoiled[kinds[test_spoiled] == kind]
        spoiled_acc = f"{(model.predict(x[spoiled]) == y[spoiled]).mean():.2f}" if len(spoiled) else "-"
        print(f"  {kind:45} {(model.predict(x[orig]) == y[orig]).mean():.2f} of {len(orig):<4} {spoiled_acc}")

    final = fit(x, y, best_c)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        weights=final.coef_.astype(np.float32),
        bias=final.intercept_.astype(np.float32),
        classes=np.array(classes),
        features=np.array(FEATURES[args.features]),
        siglip_model=data["siglip_model"],
        bioclip_model=data["bioclip_model"],
        trained_on=np.array(f"{len(np.unique(group))} photos, C={best_c}, held-back accuracy {best_acc:.3f}"),
    )
    print(f"\nsaved {args.out} (refit on all {len(y)} rows)")


if __name__ == "__main__":
    main()
