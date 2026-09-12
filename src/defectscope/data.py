"""Synthetic data and MVTec AD compatible loader with explicit split provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

CATEGORIES = ("woven", "brushed", "board")
DEFECTS = ("scratch", "stain", "hole", "swap")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)

def make_sample(category, rng, size=96, defect=None, shift=False):
    y, x = np.mgrid[:size, :size].astype(float)
    if category == "woven":
        phase = rng.uniform(-.25, .25)
        base = .47 + .08*np.sin(x*.8+phase)*np.cos(y*.8-phase) + .025*np.sin(y*1.6)
        a = np.stack([base*1.08, base, base*.84], -1)
    elif category == "brushed":
        base = .52 + .032*np.sin(y*1.8+rng.uniform(-.4, .4)) + .09*x/size
        a = np.stack([base*.91, base*.98, base*1.06], -1)
    elif category == "board":
        a = np.zeros((size, size, 3)) + [.12, .29, .23]
        traces = ((x.astype(int)%24 >= 10)&(x.astype(int)%24 <= 12)) | ((y.astype(int)%24 >= 10)&(y.astype(int)%24 <= 12))
        a[traces] = [.55, .59, .33]
        for cx, cy in [(24, 24), (72, 24), (24, 72), (72, 72)]:
            pad = (np.abs(x-cx)<7)&(np.abs(y-cy)<7)
            a[pad] = [.73, .71, .58]
    else:
        raise ValueError("unknown category")
    a += rng.normal(0, .013, a.shape) + rng.normal(0, .012)
    a *= rng.uniform(.96, 1.04)
    if shift:
        a = a*.82 + .13*(x/size)[..., None]
    mask = Image.new("L", (size, size))
    draw = ImageDraw.Draw(mask)
    cx, cy = rng.integers(18, size-18, size=2).tolist()
    if defect == "scratch":
        draw.line((cx-12, cy-9, cx+12, cy+9), fill=255, width=int(rng.integers(2, 5)))
    elif defect in ("stain", "hole"):
        radius = int(rng.integers(4, 9))
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=255)
    elif defect == "swap":
        draw.rectangle((cx-6, cy-6, cx+5, cy+5), fill=255)
    elif defect is not None:
        raise ValueError("unknown defect")
    m = np.asarray(mask) > 0
    if defect == "scratch":
        a[m] = rng.uniform(.76, .96, 3)
    elif defect == "hole":
        a[m] *= .2
    elif defect == "stain":
        a[m] = .45*a[m] + [.16, .03, .025]
    elif defect == "swap":
        # A normal-looking patch relocated: spatial branch must reason about layout.
        sx, sy = (24, 24) if category == "board" else (size-cx, size-cy)
        patch = a[sy-6:sy+6, sx-6:sx+6].copy()
        a[cy-6:cy+6, cx-6:cx+6] = patch
    return Image.fromarray(np.uint8(np.clip(a, 0, 1)*255)), mask

def generate(root, seed=7, categories=CATEGORIES):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"dataset.json").exists() or any(root.iterdir()):
        raise ValueError("output data folder must be empty; use a new folder")
    counts = {"train": 64, "tune": 24, "calibration": 39, "test": 24, "shift": 24}
    # Stable category streams; selecting a category does not change its images.
    for category in categories:
        rng = np.random.default_rng(np.random.SeedSequence([seed, CATEGORIES.index(category)]))
        for split, count in counts.items():
            for i in range(count):
                image, _ = make_sample(category, rng, shift=split=="shift")
                path = root/category/split/"good"/f"{i:03d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
        for kind in DEFECTS:
            for i in range(12):
                image, mask = make_sample(category, rng, defect=kind)
                path = root/category/"test"/kind/f"{i:03d}.png"
                target = root/category/"ground_truth"/kind/f"{i:03d}_mask.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                mask.save(target)
    (root/"dataset.json").write_text(json.dumps({"source": "procedural synthetic v1", "seed": seed, "categories": list(categories), "normal_counts": counts, "defects_per_type": 12}, indent=2), encoding="utf-8")
    return root

def splits(category_root, seed=7):
    root = Path(category_root)
    train = list_images(root/"train"/"good")
    if (root/"tune").exists() and (root/"calibration").exists():
        tune, cal = list_images(root/"tune"/"good"), list_images(root/"calibration"/"good")
    else:
        if len(train) < 20:
            raise ValueError("MVTec-style input requires >=20 train/good images")
        rng = np.random.default_rng(seed)
        ordered = [train[i] for i in rng.permutation(len(train))]
        ncal = max(4, int(len(train)*.25))
        ntune = max(4, int(len(train)*.15))
        cal, tune, train = ordered[:ncal], ordered[ncal:ncal+ntune], ordered[ncal+ntune:]
    tests = []
    for path in list_images(root/"test"):
        kind = path.parent.name
        mask = root/"ground_truth"/kind/(path.stem+"_mask.png")
        if kind != "good" and not mask.exists():
            raise ValueError("missing anomaly mask: " + str(mask))
        tests.append({"path": path, "kind": kind, "label": kind != "good", "mask": mask if kind!="good" else None})
    if not tests:
        raise ValueError("no test images")
    return train, tune, cal, tests, list_images(root/"shift"/"good")

def provenance(groups, root):
    root = Path(root).resolve()
    result, seen = {}, {}
    for name, paths in groups.items():
        rows = []
        for path in paths:
            p = Path(path).resolve()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != name:
                raise ValueError("duplicate content across splits: " + str(p))
            seen[digest] = name
            rows.append({"path": p.relative_to(root).as_posix(), "sha256": digest})
        result[name] = rows
    return result
