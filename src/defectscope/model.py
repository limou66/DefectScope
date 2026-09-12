"""Spatially gated patch memory with four separately calibrated ablations."""
import json
from pathlib import Path
import numpy as np
from .features import load_image, make_extractor
from .metrics import normal_p_value, conformal_threshold

MODES = ("global", "spatial", "fusion", "gated")

def nearest(query, bank, chunk=128):
    values, indices = [], []
    bnorm = (bank * bank).sum(1)
    for q in np.array_split(query, max(1, int(np.ceil(len(query)/chunk)))):
        dist = np.maximum((q*q).sum(1)[:, None] + bnorm[None] - 2*q @ bank.T, 0)
        idx = dist.argmin(1)
        values.append(np.sqrt(dist[np.arange(len(q)), idx] / query.shape[1]))
        indices.append(idx)
    return np.concatenate(values), np.concatenate(indices)

def coreset(features, count, seed):
    rng = np.random.default_rng(seed)
    if count < 1:
        raise ValueError("memory size must be positive")
    candidates = rng.choice(len(features), min(len(features), 4096), replace=False)
    pool = features[candidates]
    selected, closest = [], np.full(len(pool), np.inf)
    at = int(rng.integers(len(pool)))
    for _ in range(min(count, len(pool))):
        selected.append(at)
        closest = np.minimum(closest, ((pool-pool[at])**2).mean(1))
        closest[selected] = -1
        at = int(closest.argmax())
    return candidates[np.asarray(selected)]

class Detector:
    def __init__(self, extractor="texture", size=96, patch=6, memory=128, seed=7, alpha=.05, illumination=False):
        if not 0 < alpha < 1 or memory < 1:
            raise ValueError("invalid alpha or memory")
        self.config = dict(extractor=extractor, size=size, patch=patch, memory=memory, seed=seed, alpha=alpha, illumination=illumination)
        self.extractor = make_extractor(extractor, size, patch)
        self.fitted = False

    def _extract(self, path):
        image = load_image(path, self.config["size"])
        if self.config["illumination"]:
            from .illumination import align_illumination
            image = align_illumination(image, self.reference)
        return self.extractor(image)

    def _features(self, paths):
        return np.stack([self._extract(p) for p in paths])

    def fit(self, train, tune, calibration):
        groups = [list(map(Path, x)) for x in (train, tune, calibration)]
        if any(len(x) < 2 for x in groups):
            raise ValueError("each normal split needs at least two images")
        resolved = [{p.resolve() for p in x} for x in groups]
        if any(resolved[i] & resolved[j] for i in range(3) for j in range(i)):
            raise ValueError("train, tune, calibration paths must be disjoint")
        train, tune, calibration = groups
        self.reference = np.median(np.stack([load_image(p, self.config["size"]) for p in train]), axis=0).astype(np.float32)
        f = self._features(train)
        self.grid = f.shape[1]
        self.center = f.mean((0, 1, 2))
        self.scale = np.maximum(f.std((0, 1, 2)), .015)
        z = (f-self.center)/self.scale
        self.spatial_mean = z.mean(0)
        self.spatial_std = np.maximum(z.std(0), .20)
        variability = np.median(self.spatial_std, axis=-1)
        reference = max(float(np.median(variability)), .01)
        self.gate = np.clip(reference/(reference+variability), .2, .8).astype(np.float32)
        flat = z.reshape(-1, z.shape[-1])
        chosen = coreset(flat, self.config["memory"], self.config["seed"])
        self.bank = flat[chosen].astype(np.float32)
        area = self.grid*self.grid
        self.bank_sources = np.asarray([str(train[i//area].resolve()) for i in chosen])
        self.bank_positions = np.stack([(chosen % area)//self.grid, chosen % self.grid], -1)
        self.fitted = True
        components = [self._raw(x)[:2] for x in self._features(tune)]
        self.component_scale = np.maximum(np.quantile(np.stack(components), .95, axis=(0, 2, 3)), .001)
        tune_maps = {m: [] for m in MODES}
        for g, s in components:
            maps = self._combine(g, s)
            for m in MODES:
                tune_maps[m].append(maps[m])
        self.pixel_thresholds = {m: float(np.quantile(tune_maps[m], .995)) for m in MODES}
        self.calibration = {m: [] for m in MODES}
        for x in self._features(calibration):
            g, s, _ = self._raw(x)
            for m, a in self._combine(g, s).items():
                self.calibration[m].append(self.image_score(a))
        self.calibration = {m: np.asarray(v) for m, v in self.calibration.items()}
        self.split_counts = dict(train=len(train), tune=len(tune), calibration=len(calibration))
        return self

    def _raw(self, features):
        if not self.fitted:
            raise RuntimeError("fit the detector first")
        z = (features-self.center)/self.scale
        g, idx = nearest(z.reshape(-1, z.shape[-1]), self.bank)
        s = np.sqrt(np.mean(((z-self.spatial_mean)/self.spatial_std)**2, -1))
        return g.reshape(self.grid, self.grid), s, idx.reshape(self.grid, self.grid)

    def _combine(self, g, s):
        g, s = g/self.component_scale[0], s/self.component_scale[1]
        return {
            "global": g, "spatial": s, "fusion": .5*g+.5*s,
            "gated": (1-self.gate)*g+self.gate*s}

    @staticmethod
    def image_score(a):
        flat = a.ravel()
        k = max(1, int(np.ceil(len(flat)*.02)))
        return float(np.partition(flat, -k)[-k:].mean())

    def predict(self, path, mode="gated"):
        if mode not in MODES:
            raise ValueError("unknown mode")
        features = self._extract(path)
        g, s, idx = self._raw(features)
        maps = self._combine(g, s)
        a, score = maps[mode], self.image_score(maps[mode])
        p = normal_p_value(self.calibration[mode], score)
        y, x = np.unravel_index(a.argmax(), a.shape)
        matched = int(idx[y, x])
        threshold = conformal_threshold(self.calibration[mode], self.config["alpha"])
        return {"score": score, "p_value": p, "is_anomaly": p <= self.config["alpha"],
                "image_threshold": threshold if np.isfinite(threshold) else None,
                "pixel_threshold": self.pixel_thresholds[mode], "mode": mode,
                "heatmap": a, "global_map": maps["global"], "spatial_map": maps["spatial"],
                "evidence": {"query_patch_yx": [int(y), int(x)],
                             "normal_image": str(self.bank_sources[matched]),
                             "normal_patch_yx": self.bank_positions[matched].tolist(),
                             "spatial_weight": float(self.gate[y, x])}}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format_version": 1, "config": self.config, "grid": self.grid,
                "pixel_thresholds": self.pixel_thresholds, "split_counts": self.split_counts}
        arrays = {name: getattr(self, name) for name in ("center", "scale", "spatial_mean", "spatial_std", "gate", "bank", "bank_sources", "bank_positions", "component_scale", "reference")}
        with path.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(meta), **arrays,
                                **{"cal_"+m: v for m, v in self.calibration.items()})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta.get("format_version") != 1:
                raise ValueError("unsupported model format")
            model = cls(**meta["config"])
            for name in ("center", "scale", "spatial_mean", "spatial_std", "gate", "bank", "bank_sources", "bank_positions", "component_scale", "reference"):
                setattr(model, name, np.zeros((model.config["size"], model.config["size"], 3), dtype=np.float32) if name == "reference" and name not in z else z[name])
            model.grid, model.pixel_thresholds, model.split_counts = meta["grid"], meta["pixel_thresholds"], meta["split_counts"]
            model.calibration = {m: z["cal_"+m] for m in MODES}
            model.fitted = True
        return model
